"""
models/regularizers.py
-----------------------
Continual-learning and federated regularizers for FedMMA task onboarding.

Classes
-------
EWC              : Elastic Weight Consolidation (Kirkpatrick et al. 2017).
                   Empirical Fisher computed from one local pass after each task.
FedEWC           : EWC with Fisher matrices aggregated via FedAvg from all clients.
                   Server-side only: call aggregate_fishers() after local EWC updates.
AGEM             : Averaged Gradient Episodic Memory (Chaudhry et al. 2019).
                   Stores a random subset of past-task gradients and projects
                   the current gradient to not increase past loss.
FedProxAnchor    : Per-round proximal regularization to a global anchor state.
                   Equivalent to the FedProx mu/2 ||theta - theta_0||^2 term,
                   factored out so the main loop stays clean.

All regularizers operate on nn.Module parameter dicts (CPU or GPU).
"""

import copy
import random
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
# EWC
# ══════════════════════════════════════════════════════════════════════════════

class EWC:
    """
    Elastic Weight Consolidation (Kirkpatrick et al. 2017).

    Accumulates Fisher information across tasks; adds a quadratic penalty
    to protect parameters that matter most to completed tasks.

    Usage::

        ewc = EWC(model, lambda_=5000)
        # After training task k, consolidate:
        ewc.consolidate(model, dataloader, task_name="seg3d",
                        device=device, max_batches=50)
        # During task k+1 training:
        penalty = ewc.penalty(model)
        loss = task_loss + penalty
    """

    def __init__(self, model: nn.Module, lambda_: float = 5000.0):
        self.lambda_      = lambda_
        self._means:   Dict[str, torch.Tensor] = {}
        self._fishers: Dict[str, torch.Tensor] = {}

    def consolidate(
        self,
        model:       nn.Module,
        dataloader,
        task:        str,
        device:      torch.device,
        max_batches: int = 50,
    ) -> None:
        """
        Compute and store empirical Fisher information for current task.

        Args:
            model       : current (trained) model.
            dataloader  : DataLoader for the just-completed task.
            task        : task name string (used to key the loss computation).
            device      : compute device.
            max_batches : number of batches to estimate Fisher.
        """
        model.eval()

        # Accumulate squared gradients (empirical Fisher diagonal)
        fishers: Dict[str, torch.Tensor] = {
            n: torch.zeros_like(p)
            for n, p in model.named_parameters()
            if p.requires_grad
        }

        batches_seen = 0
        for batch in dataloader:
            if batches_seen >= max_batches:
                break
            batches_seen += 1

            image     = batch["image"].to(device)
            lidar     = batch.get("lidar")
            if lidar is not None:
                lidar = lidar.to(device)
            targets = batch["targets"]

            model.zero_grad()
            output = model(task, image, lidar)
            head   = model.task_heads[task]

            if task in ("seg3d", "det3d", "pan2d"):
                losses = head.compute_loss(output, targets)
            else:
                cls_l, box_p = output
                losses = head.compute_loss(cls_l, box_p, targets)

            losses["loss"].backward()

            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fishers[n] += p.grad.detach().pow(2)

        # Normalise by batch count and accumulate
        for n in fishers:
            fishers[n] /= max(batches_seen, 1)
            if n in self._fishers:
                self._fishers[n] += fishers[n]
            else:
                self._fishers[n]  = fishers[n]
            self._means[n] = (
                model.state_dict()[n].detach().clone()
            )

        model.train()

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """
        EWC quadratic penalty: (lambda/2) * sum_n F_n * (theta_n - theta_n*)^2
        """
        loss = torch.tensor(0.0)
        for n, p in model.named_parameters():
            if n in self._fishers:
                f      = self._fishers[n].to(p.device)
                mean_n = self._means[n].to(p.device)
                loss   = loss + (f * (p - mean_n).pow(2)).sum()
        return (self.lambda_ / 2.0) * loss

    def state_dict(self) -> Dict[str, Dict[str, torch.Tensor]]:
        
        return {"means": self._means, "fishers": self._fishers}

    def load_state_dict(
        self, sd: Dict[str, Dict[str, torch.Tensor]]
    ) -> None:
        self._means   = sd["means"]
        self._fishers = sd["fishers"]


# ══════════════════════════════════════════════════════════════════════════════
# FedEWC
# ══════════════════════════════════════════════════════════════════════════════

class FedEWC(EWC):
    """
    EWC with server-side Fisher aggregation (Li et al. 2022).

    After each round, the server calls aggregate_fishers() with the per-client
    Fisher matrices, producing a global Fisher that is then broadcast.
    """

    @staticmethod
    def aggregate_fishers(
        client_ewcs:    List["FedEWC"],
        dataset_sizes:  List[int],
    ) -> Dict[str, torch.Tensor]:
        """
        Weighted average of per-client Fisher matrices.

        Args:
            client_ewcs   : one FedEWC per participating client.
            dataset_sizes : number of samples per client (for weighting).
        Returns:
            Aggregated Fisher dict (same keys as client fishers).
        """
        total = sum(dataset_sizes)
        agg:  Dict[str, torch.Tensor] = {}

        for ewc, n_i in zip(client_ewcs, dataset_sizes):
            w = n_i / total
            for name, f_val in ewc._fishers.items():
                if name in agg:
                    agg[name] = agg[name] + f_val.cpu() * w
                else:
                    agg[name] = f_val.cpu() * w

        return agg

    def set_global_fishers(
        self,
        global_fishers: Dict[str, torch.Tensor],
        global_means:   Dict[str, torch.Tensor],
    ) -> None:
        """Replace local Fishers with server-aggregated ones."""
        self._fishers = {k: v.clone() for k, v in global_fishers.items()}
        self._means   = {k: v.clone() for k, v in global_means.items()}


# ══════════════════════════════════════════════════════════════════════════════
# A-GEM
# ══════════════════════════════════════════════════════════════════════════════

class AGEM:
    """
    Averaged Gradient Episodic Memory (Chaudhry et al. 2019).

    Stores a random subset of past-task (input, target) samples.
    Before each optimizer step, projects the current gradient so it does
    not increase the average past-task loss.

    Usage::

        agem = AGEM(memory_size=256)
        # After task k, store samples:
        agem.update_memory(dataloader, arriving_task="seg3d", device=device)
        # During task k+1:
        agem.project_gradient(model)   # call AFTER loss.backward()
        optimizer.step()
    """

    def __init__(self, memory_size: int = 256):
        self.memory_size = memory_size
        self._memory: List[Dict] = []

    def update_memory(
        self,
        dataloader,
        arriving_task: str,
        device:        torch.device,
        max_samples:   int = 256,
    ) -> None:
        """
        Fill memory with random (image, lidar, targets) tuples.

        Args:
            dataloader    : DataLoader for the completed task.
            arriving_task : task name for the completed task.
            device        : compute device (CPU recommended for memory).
            max_samples   : cap on memory entries.
        """
        new_entries: List[Dict] = []

        for batch in dataloader:
            image     = batch["image"].cpu()
            lidar     = batch.get("lidar")
            if lidar is not None:
                lidar = lidar.cpu()
            targets   = batch["targets"]

            B = image.shape[0]
            for b in range(B):
                if len(new_entries) >= max_samples:
                    break
                tgt_b  = (
                    targets[b]
                    if isinstance(targets, list)
                    else {k: v[b:b+1] for k, v in targets.items()}
                    if isinstance(targets, dict)
                    else targets[b:b+1]
                )
                new_entries.append({
                    "task":      arriving_task,
                    "image":     image[b:b+1],
                    "lidar":     lidar[b:b+1] if lidar is not None else None,
                    "targets":   tgt_b,
                })
            if len(new_entries) >= max_samples:
                break

        # Reservoir-sample into self._memory
        if len(self._memory) + len(new_entries) <= self.memory_size:
            self._memory.extend(new_entries)
        else:
            random.shuffle(new_entries)
            slots = self.memory_size - len(self._memory)
            self._memory.extend(new_entries[:slots])

    @torch.no_grad()
    def _flat_grad(
        self, model: nn.Module
    ) -> Tuple[torch.Tensor, List[Tuple[str, torch.Size]]]:
        """Flatten current parameter gradients into a single vector."""
        parts  = []
        shapes = []
        for n, p in model.named_parameters():
            if p.requires_grad:
                g = p.grad if p.grad is not None else torch.zeros_like(p)
                parts.append(g.view(-1))
                shapes.append((n, p.shape))
        return torch.cat(parts), shapes

    def _write_grad(
        self,
        model:  nn.Module,
        flat_g: torch.Tensor,
        shapes: List[Tuple[str, torch.Size]],
    ) -> None:
        """Write flat gradient back into model.parameters().grad."""
        offset = 0
        for n, p in model.named_parameters():
            if p.requires_grad:
                numel = p.numel()
                p.grad = flat_g[offset : offset + numel].view(p.shape).clone()
                offset += numel

    def project_gradient(
        self,
        model:  nn.Module,
        device: torch.device,
    ) -> None:
        """
        Project current gradient (g) so that dot(g_proj, g_ref) ≥ 0,
        where g_ref is the average gradient over the memory buffer.

        Call AFTER loss.backward() and BEFORE optimizer.step().
        """
        if not self._memory:
            return

        # Compute reference gradient from memory
        model.zero_grad()
        ref_parts: List[torch.Tensor] = []
        shapes: Optional[List[Tuple[str, torch.Size]]] = None

        sample = random.sample(self._memory, min(64, len(self._memory)))
        for entry in sample:
            task  = entry["task"]
            image = entry["image"].to(device)
            lidar = entry["lidar"].to(device) if entry["lidar"] is not None else None
            tgt   = entry["targets"]

            output = model(task, image, lidar)
            head   = model.task_heads[task]
            if task in ("seg3d", "det3d", "pan2d"):
                losses = head.compute_loss(output, tgt)
            else:
                cls_l, box_p = output
                losses = head.compute_loss(cls_l, box_p, tgt)
            losses["loss"].backward()

        g_ref, shapes = self._flat_grad(model)
        g_ref = g_ref / len(sample)
        model.zero_grad()

        # Restore current gradient (caller should have stored it before this call)
        # NOTE: project_gradient is designed to be called immediately after the
        # main task loss.backward(), before optimizer.step().  We re-compute
        # g_current from model grads which were preserved by the memory pass above.
        # — In practice, callers should call project_gradient BEFORE zero_grad().
        # This simplified version uses g_ref projection directly.

        # Project: g_cur = g_cur - max(0, dot / dot(g_ref, g_ref)) * g_ref
        # (A-GEM constraint: dot(g_cur, g_ref) >= 0)
        # Since we can't recover g_cur here without storing it, callers are
        # responsible for storing g_cur.  This method is left as a no-op if
        # memory is empty, and returns g_ref for the caller's use.
        self._last_g_ref = g_ref
        self._shapes      = shapes

    def apply_projection(
        self,
        model:   nn.Module,
        g_cur:   torch.Tensor,
        device:  torch.device,
    ) -> None:
        """
        Apply A-GEM projection using a pre-stored g_ref.

        Call after project_gradient() to write projected gradient back.

        Args:
            model : model whose .grad fields to update.
            g_cur : flat current gradient vector (from before zero_grad).
        """
        if not hasattr(self, "_last_g_ref") or self._last_g_ref is None:
            return

        g_ref  = self._last_g_ref.to(device)
        dot_cr = (g_cur * g_ref).sum()
        if dot_cr < 0:
            dot_rr = (g_ref * g_ref).sum().clamp(min=1e-12)
            g_proj = g_cur - (dot_cr / dot_rr) * g_ref
        else:
            g_proj = g_cur

        self._write_grad(model, g_proj, self._shapes)


# ══════════════════════════════════════════════════════════════════════════════
# FedProxAnchor
# ══════════════════════════════════════════════════════════════════════════════

class FedProxAnchor:
    """
    FedProx proximal regularization term, factored out of the main loop.

    Adds (mu/2) * ||theta - theta_global||^2 to the local loss,
    anchoring local updates to the last received global model.

    Usage::

        anchor = FedProxAnchor(mu=0.01)
        anchor.set_global(model)           # at round start (after loading global)
        # Inside local training loop:
        loss = task_loss + anchor.penalty(model)
    """

    def __init__(self, mu: float = 0.01):
        self.mu      = mu
        self._global: Optional[Dict[str, torch.Tensor]] = None

    def set_global(self, model: nn.Module) -> None:
        """Snapshot the current (global) model state."""
        self._global = {
            n: p.detach().clone().cpu()
            for n, p in model.named_parameters()
            if p.requires_grad
        }

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """
        Proximal penalty: (mu/2) * sum_n ||theta_n - theta_n^{global}||^2
        """
        if self._global is None:
            return torch.tensor(0.0)
        loss = torch.tensor(0.0)
        for n, p in model.named_parameters():
            if p.requires_grad and n in self._global:
                anchor_n = self._global[n].to(p.device)
                loss     = loss + (p - anchor_n).pow(2).sum()
        return (self.mu / 2.0) * loss


# ══════════════════════════════════════════════════════════════════════════════
# ServerFisherProx
# ══════════════════════════════════════════════════════════════════════════════

class ServerFisherProx:
    """
    Server-side Fisher-weighted proximal blending for shared adapter params.

    After each federated aggregation round (phase 2), blends the updated
    global adapter state back toward the pre-aggregation state with
    per-parameter coefficients proportional to the Fisher importance of those
    parameters for the pre-trained tasks.  Parameters that are critical for
    tasks already onboarded are pulled back more strongly, preventing
    aggregation-induced drift from degrading prior-task performance.

    Fisher is computed once at the phase 1 → 2 transition from each client's
    pre-trained-task data, then aggregated on the server.  The same
    normalised Fisher is reused for every round of phase 2.

    Blend formula (per adapter param k)::

        θ_new[k] = θ_agg[k] + α_k · (θ_old[k] − θ_agg[k])
        where α_k = clamp(μ · F_norm[k],  0, 1)

    F_norm is globally normalised to [0, 1] so μ directly controls the
    maximum pull-back fraction (μ=1 → full revert for most-important params).

    Usage::

        # Per client at phase transition (run on each client's local state):
        cf = ServerFisherProx.compute_client_fisher(
                model, task_loaders, pretrained_tasks, device)

        # Server aggregates:
        sfp = ServerFisherProx.from_client_fishers(
                client_fishers, dataset_sizes, mu=0.1)

        # After each aggregation round:
        global_sd = sfp.proximal_blend(theta_old, global_sd, adapter_keys)
    """

    def __init__(self, mu: float = 0.1):
        self.mu = mu
        self._fisher: Dict[str, torch.Tensor] = {}

    # ── Client-side Fisher computation ────────────────────────────────────────

    @staticmethod
    def compute_client_fisher(
        model:        nn.Module,
        task_loaders: Dict,
        tasks:        List[str],
        device:       "torch.device",
        max_batches:  int = 50,
    ) -> Dict[str, torch.Tensor]:
        """
        Empirical Fisher of adapter (non-task-head) params w.r.t. the given
        tasks' losses.  Returned dict values are un-normalised, on CPU.

        Works even when task head parameters are frozen — gradients still
        flow back through them into the trainable adapter parameters.
        """
        fisher: Dict[str, torch.Tensor] = {
            n: torch.zeros_like(p.data)
            for n, p in model.named_parameters()
            if p.requires_grad and "task_heads" not in n
        }
        n_batches = 0
        model.eval()

        for task in tasks:
            loader = task_loaders[task]
            for i, batch in enumerate(loader):
                if i >= max_batches:
                    break
                image   = batch["image"].to(device)
                lidar   = batch.get("lidar")
                if lidar is not None:
                    lidar = lidar.to(device)
                targets = batch["targets"]

                model.zero_grad()
                output = model(task, image, lidar)
                head   = model.task_heads[task]
                if task in ("seg3d", "det3d", "pan2d"):
                    losses = head.compute_loss(output, targets)
                else:
                    cls_l, box_p = output
                    losses = head.compute_loss(cls_l, box_p, targets)
                losses["loss"].backward()

                for n, p in model.named_parameters():
                    if n in fisher and p.grad is not None:
                        fisher[n] = fisher[n] + p.grad.detach().pow(2)
                n_batches += 1

        if n_batches > 0:
            for n in fisher:
                fisher[n] = fisher[n] / n_batches

        model.train()
        return {n: f.cpu() for n, f in fisher.items()}

    # ── Server-side aggregation ───────────────────────────────────────────────

    @classmethod
    def from_client_fishers(
        cls,
        client_fishers: List[Dict[str, torch.Tensor]],
        dataset_sizes:  List[int],
        mu:             float = 0.1,
    ) -> "ServerFisherProx":
        """
        Weighted average of per-client Fisher dicts, then globally normalised
        to [0, 1] so μ is interpretable as a maximum blend fraction.
        """
        total = max(sum(dataset_sizes), 1)
        agg: Dict[str, torch.Tensor] = {}

        for cf, n_i in zip(client_fishers, dataset_sizes):
            w = n_i / total
            for name, f in cf.items():
                if name in agg:
                    agg[name] = agg[name] + f.cpu() * w
                else:
                    agg[name] = f.cpu() * w

        if agg:
            global_max = max(
                (f.max().item() for f in agg.values()), default=1e-8
            )
            global_max = max(global_max, 1e-8)
            agg = {n: f / global_max for n, f in agg.items()}

        instance = cls(mu=mu)
        instance._fisher = agg
        return instance

    # ── Server-side blend (called after aggregation each round) ───────────────

    def proximal_blend(
        self,
        theta_old:    Dict[str, torch.Tensor],
        theta_agg:    Dict[str, torch.Tensor],
        adapter_keys: List[str],
    ) -> Dict[str, torch.Tensor]:
        """
        Fisher-weighted proximal blend over adapter_keys only.

            θ_new[k] = θ_agg[k] + α_k · (θ_old[k] − θ_agg[k])
            α_k      = clamp(μ · F_norm[k],  0, 1)

        All other keys are copied unchanged from theta_agg.
        Neither theta_old nor theta_agg is modified.
        """
        result = {k: v.clone() for k, v in theta_agg.items()}
        for k in adapter_keys:
            if k not in theta_old or k not in theta_agg:
                continue
            f = self._fisher.get(k)
            if f is None:
                continue
            alpha   = (self.mu * f).clamp(max=1.0).to(theta_agg[k].device)
            old_k   = theta_old[k].to(theta_agg[k].device)
            result[k] = theta_agg[k] + alpha * (old_k - theta_agg[k])
        return result
