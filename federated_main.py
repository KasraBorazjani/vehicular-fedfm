"""
federated_main.py
-----------------
Standalone driver for FedMMA multi-task federated learning on Waymo Open Dataset.

Baselines
---------
  fedmma     — FedMMA: global adapters aggregated, task heads local  [proposed]
  fedavg     — FedAvg: all trainable params aggregated each round
  fedprox    — FedAvg + proximal regularization (mu * ||theta-theta_0||^2) every round
  local      — No aggregation; each client trains independently
  sequential — Centralised, arriving task only  (all data, no clients)
  multitask  — Centralised, all 4 tasks jointly

Training phases
---------------
  Phase 1 (--phase1-rounds):
      All adapters + task heads trainable.  Gradient-conflict mitigation:
        --phase1-reg none      (default) plain loss sum
        --phase1-reg cagrad    CAGrad Frank-Wolfe on shared params
        --phase1-reg gradnorm  GradNorm dynamic task weighting

  Phase 2 (--phase2-rounds):
      Lidar stem is frozen.  Two independent axes of control:

      --train-pretrained-heads  (multi-task fine-tune paradigm):
          All four task heads remain trainable.
          --phase2-reg choices: none | cagrad | gradnorm

      (default, no flag)  continual-learning paradigm:
          Only the arriving task head is trained; pretrained heads frozen.
          --phase2-reg choices: none | ewc | agem | fedewc

      --phase2-prox  (server-side Fisher-weighted proximal blend):
          After each aggregation round, blends updated adapter params toward
          the pre-aggregation global state, weighted by pre-trained tasks'
          Fisher importance.  Compatible with any --phase2-reg choice in
          either paradigm.  Requires fedmma / fedavg / fedprox baseline.

  fedewc is only valid for federated baselines (fedmma/fedavg/fedprox/local).
  Combining --baseline fedprox with a CL phase2-reg is not supported.

Usage
-----
  # CL paradigm, EWC in phase 2
  python federated_main.py \\
      --baseline fedmma --arriving-task seg3d \\
      --pt-dir /path/to/waymo_pt --output-dir output/fedmma_seg3d \\
      --phase1-reg none --phase2-reg ewc

  # Multi-task fine-tune paradigm, CAGrad in phase 1, GradNorm in phase 2
  python federated_main.py \\
      --baseline fedavg --arriving-task det3d --train-pretrained-heads \\
      --pt-dir /path/to/waymo_pt --output-dir output/fedavg_det3d \\
      --phase1-reg cagrad --phase2-reg gradnorm

SLURM job array: scripts/fedmma_array.slurm
"""

import argparse
import copy
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

from clip import clip
from models.custom_clip import CustomCLIP
from utils.text_templates import TASKS, TASK_CLASSES
from utils.fed_utils import fedavg_aggregate, fedmma_aggregate, compute_miou, compute_map50


# ── Constants ─────────────────────────────────────────────────────────────────

NUM_CLASSES = {
    "seg3d": len(TASK_CLASSES["seg3d"]),   # 22
    "det3d": len(TASK_CLASSES["det3d"]),   # 4
    "pan2d": len(TASK_CLASSES["pan2d"]),   # 28
    "det2d": len(TASK_CLASSES["det2d"]),   # 3
}

CENTRALIZED_BASELINES = {"sequential", "multitask"}

# Methods valid only when train_pretrained_heads=True  (multi-task fine-tune paradigm)
_MTL_REGS = {"cagrad", "gradnorm"}
# Methods valid only when train_pretrained_heads=False  (continual-learning paradigm)
_CL_REGS  = {"ewc", "agem", "fedewc"}


# ══════════════════════════════════════════════════════════════════════════════
# Argument validation
# ══════════════════════════════════════════════════════════════════════════════

def _validate_args(args) -> None:
    p2 = args.phase2_reg

    if p2 in _MTL_REGS and not args.train_pretrained_heads:
        raise SystemExit(
            f"ERROR: --phase2-reg {p2} requires --train-pretrained-heads "
            f"(it is a multi-task fine-tune regularizer, not a CL regularizer)."
        )
    if p2 in _CL_REGS and args.train_pretrained_heads:
        raise SystemExit(
            f"ERROR: --phase2-reg {p2} is a CL regularizer; "
            f"do not pass --train-pretrained-heads."
        )
    if p2 == "fedewc" and args.baseline in CENTRALIZED_BASELINES:
        raise SystemExit(
            "ERROR: --phase2-reg fedewc requires a federated baseline "
            "(fedmma / fedavg / fedprox / local)."
        )
    if args.baseline == "fedprox" and p2 in _CL_REGS:
        raise SystemExit(
            f"ERROR: combining --baseline fedprox (proximal penalty every round) "
            f"with --phase2-reg {p2} (CL regularizer) is not supported. "
            f"Use --baseline fedavg --phase2-reg {p2} instead."
        )
    if args.phase2_prox and args.baseline in {*CENTRALIZED_BASELINES, "local"}:
        raise SystemExit(
            f"ERROR: --phase2-prox requires an aggregating federated baseline "
            f"(fedmma / fedavg / fedprox). '{args.baseline}' has no cross-client "
            f"aggregation step to apply it at."
        )
    if args.phase1_reg != "none" and args.baseline == "sequential":
        print(
            f"WARNING: --phase1-reg {args.phase1_reg} has no effect for 'sequential' "
            f"(single-task baseline; no gradient conflict to resolve)."
        )


# ══════════════════════════════════════════════════════════════════════════════
# Model construction
# ══════════════════════════════════════════════════════════════════════════════

def _load_clip_cpu(backbone: str = "ViT-B/16") -> nn.Module:
    url        = clip._MODELS[backbone]
    model_path = clip._download(url)
    try:
        m  = torch.jit.load(model_path, map_location="cpu").eval()
        sd = None
    except RuntimeError:
        sd = torch.load(model_path, map_location="cpu")
        m  = None
    from clip.model_mma import build_model_MMA
    return build_model_MMA(sd or m.state_dict())


def build_model(args) -> CustomCLIP:
    clip_model = _load_clip_cpu(args.backbone)
    clip_model.float()
    return CustomCLIP(
        clip_model              = clip_model,
        arriving_task           = args.arriving_task,
        adapter_dim             = args.adapter_dim,
        shared_adapter_dim      = args.shared_adapter_dim,
        adapter_start           = args.adapter_start,
        adapter_scale           = args.adapter_scale,
        num_decoder_layers      = args.num_decoder_layers,
        num_queries_3d          = args.num_queries_3d,
        num_queries_2d          = args.num_queries_2d,
        num_queries_pan         = args.num_queries_pan,
        use_task_adapter        = args.use_task_adapter,
        freeze_pretrained_heads = not args.train_pretrained_heads,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Key partitioning for FL aggregation
# ══════════════════════════════════════════════════════════════════════════════

def _global_keys(model: CustomCLIP, train_pretrained_heads: bool = False) -> List[str]:
    """
    Keys aggregated globally in FedMMA.

    When train_pretrained_heads=True, pretrained (non-arriving) head keys are
    also aggregated globally — they are shared across clients and should
    converge to the same values, so global aggregation is appropriate.
    """
    base = [k for k in model.state_dict() if any(
        s in k for s in [
            "vis_adapters", "txt_adapters", "lidar_encoder",
            "image_lidar_fusion", "task_adapter",
        ]
    )]
    if train_pretrained_heads:
        pretrained_keys = [
            k for k in model.state_dict()
            if "task_heads" in k
            and f"task_heads.{model.arriving_task}" not in k
        ]
        return base + pretrained_keys
    return base


def _local_keys(model: CustomCLIP, arriving_task: str) -> List[str]:
    """Keys kept per-client in FedMMA (arriving task head only)."""
    return [k for k in model.state_dict() if f"task_heads.{arriving_task}" in k]


def _trainable_keys(model: CustomCLIP) -> List[str]:
    return [n for n, p in model.named_parameters() if p.requires_grad]


# ══════════════════════════════════════════════════════════════════════════════
# Loss computation (unified across tasks)
# ══════════════════════════════════════════════════════════════════════════════

def compute_loss(
    model:     CustomCLIP,
    batch:     Dict,
    task:      str,
    device:    torch.device,
) -> Dict[str, torch.Tensor]:
    image  = batch["image"].to(device)
    lidar  = batch.get("lidar")
    if lidar is not None:
        lidar = lidar.to(device)
    targets = batch["targets"]

    output = model(task, image, lidar)
    head   = model.task_heads[task]

    if task in ("seg3d", "det3d", "pan2d"):
        return head.compute_loss(output, targets)
    else:   # det2d
        cls_l, box_p = output
        return head.compute_loss(cls_l, box_p, targets)


# ══════════════════════════════════════════════════════════════════════════════
# Local training — single-task
# ══════════════════════════════════════════════════════════════════════════════

def local_train(
    model:         CustomCLIP,
    loader,
    optimizer:     torch.optim.Optimizer,
    scaler:        Optional[GradScaler],
    device:        torch.device,
    arriving_task: str,
    local_epochs:  int,
    regularizer    = None,
    verbose:       bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Train model locally on a single task; return final CPU state_dict.

    regularizer may be:
      None             — no regularization
      EWC / FedEWC     — penalty(model) added to task loss before backward
      FedProxAnchor    — penalty(model) added to task loss before backward
      AGEM             — gradient projection applied after backward
    """
    from models.regularizers import AGEM as _AGEM

    is_agem = isinstance(regularizer, _AGEM)
    model.train()

    for epoch in range(local_epochs):
        epoch_loss = 0.0
        n_batches  = 0
        for batch in loader:
            optimizer.zero_grad()

            if scaler is not None:
                with autocast("cuda"):
                    losses = compute_loss(model, batch, arriving_task, device)
                    loss   = losses["loss"]
                    if regularizer is not None and not is_agem:
                        loss = loss + regularizer.penalty(model)
                scaler.scale(loss).backward()
                if is_agem:
                    # unscale so grads are in true float scale before projection
                    scaler.unscale_(optimizer)
                    g_cur, _ = regularizer._flat_grad(model)
                    regularizer.project_gradient(model, device)
                    regularizer.apply_projection(model, g_cur, device)
                scaler.step(optimizer)
                scaler.update()
            else:
                losses = compute_loss(model, batch, arriving_task, device)
                loss   = losses["loss"]
                if regularizer is not None and not is_agem:
                    loss = loss + regularizer.penalty(model)
                loss.backward()
                if is_agem:
                    g_cur, _ = regularizer._flat_grad(model)
                    regularizer.project_gradient(model, device)
                    regularizer.apply_projection(model, g_cur, device)
                optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        if verbose and n_batches:
            print(f"      epoch {epoch+1}/{local_epochs}  "
                  f"loss={epoch_loss/n_batches:.4f}")

    return {k: v.cpu() for k, v in model.state_dict().items()}


# ══════════════════════════════════════════════════════════════════════════════
# Multi-task step helpers  (shared by federated + centralised paths)
# ══════════════════════════════════════════════════════════════════════════════

def _standard_multitask_step(
    model, iterators, tasks, optimizer, scaler, device,
    gradnorm=None, regularizer=None,
):
    """One step: one mini-batch per task, summed (or GradNorm-weighted) loss."""
    optimizer.zero_grad()
    task_losses = []

    for t in tasks:
        try:
            batch = next(iterators[t])
        except StopIteration:
            continue
        if scaler:
            with autocast("cuda"):
                losses = compute_loss(model, batch, t, device)
        else:
            losses = compute_loss(model, batch, t, device)
        task_losses.append(losses["loss"])

    if not task_losses:
        return

    task_loss_tensor = torch.stack(task_losses)
    if gradnorm is not None:
        total, _ = gradnorm.weighted_loss(task_loss_tensor)
    else:
        total = task_loss_tensor.sum()

    if regularizer is not None:
        total = total + regularizer.penalty(model)

    if scaler:
        scaler.scale(total).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        total.backward()
        optimizer.step()

    if gradnorm is not None:
        shared_p = next((p for p in model.vis_adapters.parameters()), None)
        if shared_p is not None:
            gradnorm.step(shared_p, task_loss_tensor.detach())


def _cagrad_step(
    model, iterators, tasks, optimizer, scaler, device, alpha,
    regularizer=None,
):
    """
    One CAGrad step: per-task gradients accumulated on shared params then combined.

    regularizer (optional FedProxAnchor): its gradient is added on top of the
    CAGrad-combined shared gradient before the optimizer step.
    """
    from utils.fed_utils import cagrad as _cagrad

    shared_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and "task_heads" not in n
    ]

    per_task_grads: List[torch.Tensor] = []

    for t in tasks:
        try:
            batch = next(iterators[t])
        except StopIteration:
            continue

        optimizer.zero_grad()
        if scaler:
            with autocast("cuda"):
                losses = compute_loss(model, batch, t, device)
            scaler.scale(losses["loss"]).backward()
        else:
            losses = compute_loss(model, batch, t, device)
            losses["loss"].backward()

        flat_g = torch.cat([
            p.grad.view(-1) if p.grad is not None
            else torch.zeros(p.numel(), device=device)
            for p in shared_params
        ])
        per_task_grads.append(flat_g.detach())

    if not per_task_grads:
        return

    combined_g = _cagrad(torch.stack(per_task_grads), alpha)

    optimizer.zero_grad()
    offset = 0
    for p in shared_params:
        n      = p.numel()
        p.grad = combined_g[offset : offset + n].view(p.shape).clone()
        offset += n

    # Add regularizer gradient on top (e.g. FedProxAnchor for task-head params)
    if regularizer is not None:
        reg_loss = regularizer.penalty(model)
        reg_loss.backward()

    if scaler:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()


def _multitask_local_train(
    model,
    task_loaders:  Dict,
    optimizer:     torch.optim.Optimizer,
    scaler:        Optional[GradScaler],
    device:        torch.device,
    tasks:         List[str],
    local_epochs:  int,
    phase_reg:     str,
    gradnorm       = None,
    regularizer    = None,
    cagrad_alpha:  float = 0.5,
) -> Dict[str, torch.Tensor]:
    """
    Multi-task local training for federated clients (and centralised when reused).

    phase_reg  : "none" | "cagrad" | "gradnorm"  — gradient-conflict method
    gradnorm   : GradNorm instance when phase_reg == "gradnorm"
    regularizer: FedProxAnchor (from --baseline fedprox); always added as a
                 penalty term on top of whatever gradient method is active
    """
    model.train()
    for _ in range(local_epochs):
        iterators = {t: iter(task_loaders[t]) for t in tasks}
        if phase_reg == "cagrad":
            _cagrad_step(
                model, iterators, tasks, optimizer, scaler, device,
                cagrad_alpha, regularizer=regularizer,
            )
        elif phase_reg == "gradnorm":
            _standard_multitask_step(
                model, iterators, tasks, optimizer, scaler, device,
                gradnorm=gradnorm, regularizer=regularizer,
            )
        else:  # "none"
            _standard_multitask_step(
                model, iterators, tasks, optimizer, scaler, device,
                gradnorm=None, regularizer=regularizer,
            )
    return {k: v.cpu() for k, v in model.state_dict().items()}


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(
    model:         CustomCLIP,
    loader,
    arriving_task: str,
    device:        torch.device,
    max_batches:   int = 100,
) -> Dict[str, float]:
    model.eval()

    if arriving_task == "seg3d":
        all_preds, all_labels = [], []
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            image  = batch["image"].to(device)
            lidar  = batch.get("lidar")
            if lidar is not None:
                lidar = lidar.to(device)
            output = model("seg3d", image, lidar)
            all_preds.append(output.cpu())
            all_labels.append(batch["targets"].cpu())

        if not all_preds:
            return {"miou": 0.0}
        return compute_miou(
            torch.cat(all_preds, 0), torch.cat(all_labels, 0), num_classes=22
        )

    elif arriving_task == "det2d":
        all_pb, all_sc, all_pl, all_gb, all_gl = [], [], [], [], []
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            image        = batch["image"].to(device)
            cls_l, box_p = model("det2d", image, None)
            scores, labels = cls_l[..., :3].max(-1)
            for b in range(image.shape[0]):
                tgt = batch["targets"][b]
                all_pb.append(box_p[b].cpu());  all_sc.append(scores[b].cpu())
                all_pl.append(labels[b].cpu())
                all_gb.append(tgt["boxes"].cpu()); all_gl.append(tgt["labels"].cpu())

        if not all_pb:
            return {"map50": 0.0}
        return compute_map50(all_pb, all_sc, all_pl, all_gb, all_gl, num_classes=3)

    return {"metric": 0.0}


# ══════════════════════════════════════════════════════════════════════════════
# Checkpointing
# ══════════════════════════════════════════════════════════════════════════════

def save_checkpoint(
    output_dir: str,
    round_idx:  int,
    phase:      int,
    global_sd:  Dict,
    local_sds:  Optional[List[Dict]],
    metrics:    Dict,
) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ckpt = {
        "round":     round_idx,
        "phase":     phase,
        "global_sd": global_sd,
        "local_sds": local_sds,
        "metrics":   metrics,
    }
    path = os.path.join(output_dir, f"ckpt_phase{phase}_r{round_idx:04d}.pt")
    torch.save(ckpt, path)


def load_latest_checkpoint(output_dir: str) -> Optional[Dict]:
    pts = sorted(Path(output_dir).glob("ckpt_phase*.pt"))
    return torch.load(pts[-1], map_location="cpu") if pts else None


# ══════════════════════════════════════════════════════════════════════════════
# CL-regularizer consolidation  (called at phase 1 → 2 transition)
# ══════════════════════════════════════════════════════════════════════════════

def _consolidate_cl_regularizers(
    args,
    model:      CustomCLIP,
    mgr,
    global_sd:  Dict,
    local_sds:  List[Dict],
    device:     torch.device,
) -> List:
    """
    Build one CL regularizer per client from arriving-task data.
    For FedEWC, the caller is responsible for server-side Fisher aggregation.
    """
    from models.regularizers import EWC, FedEWC, AGEM

    regularizers = []
    for c_id in range(args.num_clients):
        # Reconstitute this client's model state
        init_sd = copy.deepcopy(global_sd)
        for k, v in local_sds[c_id].items():
            init_sd[k] = v
        model.load_state_dict(init_sd, strict=False)
        model.eval()

        loader = mgr.get_loader(c_id, args.arriving_task)

        if args.phase2_reg == "ewc":
            reg = EWC(model, lambda_=args.ewc_lambda)
            reg.consolidate(model, loader, task=args.arriving_task,
                            device=device, max_batches=args.ewc_max_batches)
        elif args.phase2_reg == "fedewc":
            reg = FedEWC(model, lambda_=args.ewc_lambda)
            reg.consolidate(model, loader, task=args.arriving_task,
                            device=device, max_batches=args.ewc_max_batches)
        elif args.phase2_reg == "agem":
            reg = AGEM(memory_size=args.agem_memory_size)
            reg.update_memory(loader, args.arriving_task,
                              device=device, max_samples=args.agem_memory_size)
        else:
            reg = None

        regularizers.append(reg)

    model.train()
    return regularizers


# ══════════════════════════════════════════════════════════════════════════════
# Federated training loop
# ══════════════════════════════════════════════════════════════════════════════

def run_federated(args, model: CustomCLIP, device: torch.device):
    """FL baselines: fedmma / fedavg / fedprox / local."""
    from utils.waymo_data_manager import WaymoDataManager

    mgr = WaymoDataManager(
        pt_dir        = args.pt_dir,
        num_clients   = args.num_clients,
        arriving_task = args.arriving_task,
        batch_size    = args.batch_size,
        pan_seq_len   = args.pan_seq_len,
        num_workers   = args.num_workers,
        seed          = args.seed,
        eval_frac     = args.eval_frac,
    )

    global_sd  = {k: v.cpu() for k, v in model.state_dict().items()}
    local_sds  = [copy.deepcopy(global_sd) for _ in range(args.num_clients)]

    g_keys = _global_keys(model, args.train_pretrained_heads)
    l_keys = _local_keys(model, args.arriving_task)
    t_keys = _trainable_keys(model)

    # FedProx baseline: proximal anchor applied every round (all phases)
    baseline_anchor = None
    if args.baseline == "fedprox":
        from models.regularizers import FedProxAnchor
        baseline_anchor = FedProxAnchor(mu=args.mu)

    # Per-client CL regularizers (populated at phase transition if needed)
    client_regularizers: Optional[List] = None

    metrics_log:  List[Dict] = []
    eval_loader   = mgr.get_eval_loader(args.arriving_task)
    client_sizes  = mgr.arriving_task_sizes()

    # server_prox is built at phase transition if --phase2-prox is set
    server_prox = None

    global_round = 0
    for phase, n_rounds, freeze_lidar in [
        (1, args.phase1_rounds, False),
        (2, args.phase2_rounds, True),
    ]:
        if freeze_lidar:
            # ── Phase 1 → 2 transition ─────────────────────────────────────────
            model.load_state_dict(global_sd, strict=False)
            model.freeze_lidar_stem()
            global_sd = {k: v.cpu() for k, v in model.state_dict().items()}
            t_keys    = _trainable_keys(model)
            g_keys    = _global_keys(model, args.train_pretrained_heads)
            print("\n── Phase 2: lidar stem frozen ──")

            # Consolidate CL regularizers from arriving-task data
            if args.phase2_reg in _CL_REGS:
                print(f"  Consolidating {args.phase2_reg.upper()} from arriving-task data …")
                client_regularizers = _consolidate_cl_regularizers(
                    args, model, mgr, global_sd, local_sds, device,
                )

                if args.phase2_reg == "fedewc":
                    from models.regularizers import FedEWC
                    sizes = [int(client_sizes[c]) for c in range(args.num_clients)]
                    global_fishers = FedEWC.aggregate_fishers(client_regularizers, sizes)
                    # Snapshot global means from current model
                    global_means = {
                        n: p.detach().cpu()
                        for n, p in model.named_parameters()
                        if p.requires_grad
                    }
                    for ewc in client_regularizers:
                        ewc.set_global_fishers(global_fishers, global_means)

            # Server-side Fisher-weighted proximal blend: compute once at transition
            if args.phase2_prox:
                from models.regularizers import ServerFisherProx
                pretrained_tasks = [t for t in TASKS if t != args.arriving_task]
                if pretrained_tasks:
                    print("  Computing pre-trained task Fisher for server-side proximal blend …")
                    client_fishers = []
                    for c_id in range(args.num_clients):
                        init_sd = copy.deepcopy(global_sd)
                        for k, v in local_sds[c_id].items():
                            init_sd[k] = v
                        model.load_state_dict(init_sd, strict=False)
                        task_loaders = {t: mgr.get_loader(c_id, t) for t in pretrained_tasks}
                        cf = ServerFisherProx.compute_client_fisher(
                            model, task_loaders, pretrained_tasks,
                            device, max_batches=args.ewc_max_batches,
                        )
                        client_fishers.append(cf)
                    sizes = [int(client_sizes[c]) for c in range(args.num_clients)]
                    server_prox = ServerFisherProx.from_client_fishers(
                        client_fishers, sizes, mu=args.mu,
                    )
                    print(f"  ServerFisherProx built (μ={args.mu}).")

        # Build phase-level GradNorm instance (one per phase, shared across rounds)
        active_reg  = args.phase1_reg if phase == 1 else args.phase2_reg
        phase_gradnorm = None
        if active_reg == "gradnorm":
            from utils.fed_utils import GradNorm
            phase_gradnorm = GradNorm(
                num_tasks=len(TASKS), alpha=args.gradnorm_alpha,
            )

        # In phase 2 with train_pretrained_heads, multi-task even with phase2_reg="none"
        phase2_multitask = (phase == 2 and args.train_pretrained_heads)

        print(f"\n── Phase {phase}: {n_rounds} rounds ──")

        for _ in range(n_rounds):
            round_start = time.time()

            m       = max(1, int(args.frac * args.num_clients))
            sampled = random.sample(range(args.num_clients), m)
            local_upd: List[Dict]   = []
            w_sampled:  List[float] = []

            for c_id in sampled:
                init_sd = copy.deepcopy(global_sd)
                if args.baseline == "fedmma":
                    init_sd.update({k: local_sds[c_id][k] for k in l_keys
                                    if k in local_sds[c_id]})
                model.load_state_dict(init_sd, strict=False)

                optimizer = torch.optim.AdamW(
                    [p for p in model.parameters() if p.requires_grad],
                    lr=args.lr, weight_decay=args.weight_decay,
                )
                scaler = GradScaler("cuda") if args.prec == "amp" else None

                # ── Determine training path for this client/phase ──────────────
                use_multitask = (
                    phase2_multitask
                    or (phase == 1 and args.phase1_reg in ("cagrad", "gradnorm"))
                )

                if baseline_anchor is not None:
                    baseline_anchor.set_global(model)

                if use_multitask:
                    task_loaders = {t: mgr.get_loader(c_id, t) for t in TASKS}
                    final_sd = _multitask_local_train(
                        model, task_loaders, optimizer, scaler, device,
                        tasks        = TASKS,
                        local_epochs = args.local_epochs,
                        phase_reg    = active_reg if active_reg in _MTL_REGS else "none",
                        gradnorm     = phase_gradnorm if active_reg == "gradnorm" else None,
                        regularizer  = baseline_anchor,
                        cagrad_alpha = args.cagrad_alpha,
                    )
                else:
                    regularizer = (
                        baseline_anchor if baseline_anchor is not None
                        else client_regularizers[c_id] if client_regularizers is not None
                        else None
                    )
                    loader = mgr.get_loader(c_id, args.arriving_task)
                    final_sd = local_train(
                        model, loader, optimizer, scaler, device,
                        arriving_task = args.arriving_task,
                        local_epochs  = args.local_epochs,
                        regularizer   = regularizer,
                    )

                local_upd.append(final_sd)
                w_sampled.append(float(client_sizes[c_id]))

            # ── Aggregation ────────────────────────────────────────────────────
            # Save pre-aggregation state for server-side proximal blend
            theta_old = (
                copy.deepcopy(global_sd)
                if server_prox is not None and phase == 2
                else None
            )

            if args.baseline == "local":
                for i, c_id in enumerate(sampled):
                    local_sds[c_id] = local_upd[i]
                global_sd = local_upd[0]

            elif args.baseline in ("fedavg", "fedprox"):
                global_sd = fedavg_aggregate(local_upd, w_sampled, keys=t_keys)
                for c_id in range(args.num_clients):
                    local_sds[c_id] = copy.deepcopy(global_sd)

            elif args.baseline == "fedmma":
                new_global = fedmma_aggregate(local_upd, g_keys, w_sampled)
                global_sd.update(new_global)
                for i, c_id in enumerate(sampled):
                    for k in l_keys:
                        if k in local_upd[i]:
                            local_sds[c_id][k] = local_upd[i][k]

            # ── Server-side Fisher-weighted proximal blend ─────────────────────
            if theta_old is not None and args.baseline != "local":
                global_sd = server_prox.proximal_blend(theta_old, global_sd, g_keys)

            # ── Evaluation ─────────────────────────────────────────────────────
            row: Dict = {"phase": phase, "round": global_round}
            if (global_round + 1) % args.eval_interval == 0:
                eval_sd = copy.deepcopy(global_sd)
                if args.baseline == "fedmma" and sampled:
                    c0 = sampled[0]
                    eval_sd.update({k: local_sds[c0][k] for k in l_keys
                                    if k in local_sds[c0]})
                model.load_state_dict(eval_sd, strict=False)
                metrics = evaluate(model, eval_loader, args.arriving_task, device)
                row.update(metrics)
                elapsed = time.time() - round_start
                key_m   = next(iter(metrics))
                print(f"  Phase {phase} | R{global_round:4d} | "
                      f"{key_m}={metrics[key_m]:.4f} | {elapsed:.1f}s")

            metrics_log.append(row)
            global_round += 1

        save_checkpoint(
            args.output_dir, global_round - 1, phase,
            global_sd, local_sds, metrics_log[-1],
        )

    return global_sd, local_sds, metrics_log


# ══════════════════════════════════════════════════════════════════════════════
# Centralised baselines
# ══════════════════════════════════════════════════════════════════════════════

def run_centralised(args, model: CustomCLIP, device: torch.device):
    """Sequential and multitask centralised baselines."""
    all_segs = sorted(
        os.path.join(args.pt_dir, f)
        for f in os.listdir(args.pt_dir) if f.endswith(".pt")
    )
    if not all_segs:
        raise ValueError(f"No .pt files in {args.pt_dir!r}")

    rng = random.Random(args.seed)
    rng.shuffle(all_segs)
    n_eval     = max(1, int(len(all_segs) * args.eval_frac))
    eval_segs  = all_segs[:n_eval]
    train_segs = all_segs[n_eval:]

    from datasets.waymo import WaymoMultiTaskDataset, waymo_collate_fn
    from torch.utils.data import DataLoader

    train_ds = WaymoMultiTaskDataset(train_segs, pan_seq_len=args.pan_seq_len)
    eval_ds  = WaymoMultiTaskDataset(eval_segs,  pan_seq_len=args.pan_seq_len)

    def _loader(ds, task, shuffle=True):
        bs = (max(1, args.batch_size // args.pan_seq_len)
              if task == "pan2d" else args.batch_size)
        return DataLoader(
            ds.task_dataset(task),
            batch_size=bs, shuffle=shuffle,
            num_workers=args.num_workers, collate_fn=waymo_collate_fn,
            drop_last=True, pin_memory=True,
        )

    train_tasks   = [args.arriving_task] if args.baseline == "sequential" else TASKS
    eval_loader   = _loader(eval_ds, args.arriving_task, shuffle=False)
    train_loaders = {t: _loader(train_ds, t) for t in train_tasks}

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scaler = GradScaler("cuda") if args.prec == "amp" else None

    # CL regularizer for sequential baseline (populated at phase transition)
    seq_regularizer = None

    metrics_log: List[Dict] = []
    global_round = 0

    for phase, n_rounds, freeze_lidar in [
        (1, args.phase1_rounds, False),
        (2, args.phase2_rounds, True),
    ]:
        if freeze_lidar:
            model.freeze_lidar_stem()
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=args.lr, weight_decay=args.weight_decay,
            )
            print("\n── Phase 2: lidar stem frozen ──")

            # Consolidate CL regularizer for sequential baseline
            if args.baseline == "sequential" and args.phase2_reg in _CL_REGS:
                from models.regularizers import EWC, AGEM
                print(f"  Consolidating {args.phase2_reg.upper()} (centralised) …")
                loader_for_consol = _loader(train_ds, args.arriving_task)
                if args.phase2_reg == "ewc":
                    seq_regularizer = EWC(model, lambda_=args.ewc_lambda)
                    seq_regularizer.consolidate(
                        model, loader_for_consol, task=args.arriving_task,
                        device=device, max_batches=args.ewc_max_batches,
                    )
                elif args.phase2_reg == "agem":
                    seq_regularizer = AGEM(memory_size=args.agem_memory_size)
                    seq_regularizer.update_memory(
                        loader_for_consol, args.arriving_task,
                        device=device, max_samples=args.agem_memory_size,
                    )
                # fedewc blocked for centralised by _validate_args

        # Build phase-level GradNorm (persists across rounds within phase)
        active_reg     = args.phase1_reg if phase == 1 else args.phase2_reg
        phase_gradnorm = None
        if active_reg == "gradnorm" and args.baseline == "multitask":
            from utils.fed_utils import GradNorm
            phase_gradnorm = GradNorm(
                num_tasks=len(train_tasks), alpha=args.gradnorm_alpha,
            )

        print(f"\n── Phase {phase}: {n_rounds} rounds (centralised) ──")
        iterators = {t: iter(train_loaders[t]) for t in train_tasks}

        for _ in range(n_rounds):
            round_start = time.time()
            model.train()

            if args.baseline == "sequential":
                _sequential_step(
                    model, train_loaders[args.arriving_task],
                    optimizer, scaler, device, args,
                    regularizer=seq_regularizer if phase == 2 else None,
                )
            else:
                # multitask
                if active_reg == "cagrad":
                    _cagrad_step(
                        model, iterators, train_tasks, optimizer, scaler,
                        device, args.cagrad_alpha,
                    )
                else:
                    _standard_multitask_step(
                        model, iterators, train_tasks, optimizer, scaler,
                        device,
                        gradnorm=phase_gradnorm if active_reg == "gradnorm" else None,
                    )

            # Refresh iterators each round
            for t in train_tasks:
                iterators[t] = iter(train_loaders[t])

            row: Dict = {"phase": phase, "round": global_round}
            if (global_round + 1) % args.eval_interval == 0:
                metrics = evaluate(model, eval_loader, args.arriving_task, device)
                row.update(metrics)
                elapsed = time.time() - round_start
                key_m   = next(iter(metrics))
                print(f"  Phase {phase} | R{global_round:4d} | "
                      f"{key_m}={metrics[key_m]:.4f} | {elapsed:.1f}s")

            metrics_log.append(row)
            global_round += 1

        save_checkpoint(
            args.output_dir, global_round - 1, phase,
            {k: v.cpu() for k, v in model.state_dict().items()},
            None, metrics_log[-1],
        )

    return metrics_log


def _sequential_step(model, loader, optimizer, scaler, device, args, regularizer=None):
    """One epoch over the arriving-task loader (sequential centralised baseline)."""
    from models.regularizers import AGEM as _AGEM
    is_agem = isinstance(regularizer, _AGEM)

    for _ in range(args.local_epochs):
        for batch in loader:
            optimizer.zero_grad()
            if scaler:
                with autocast("cuda"):
                    losses = compute_loss(model, batch, args.arriving_task, device)
                    loss   = losses["loss"]
                    if regularizer is not None and not is_agem:
                        loss = loss + regularizer.penalty(model)
                scaler.scale(loss).backward()
                if is_agem:
                    scaler.unscale_(optimizer)
                    g_cur, _ = regularizer._flat_grad(model)
                    regularizer.project_gradient(model, device)
                    regularizer.apply_projection(model, g_cur, device)
                scaler.step(optimizer)
                scaler.update()
            else:
                losses = compute_loss(model, batch, args.arriving_task, device)
                loss   = losses["loss"]
                if regularizer is not None and not is_agem:
                    loss = loss + regularizer.penalty(model)
                loss.backward()
                if is_agem:
                    g_cur, _ = regularizer._flat_grad(model)
                    regularizer.project_gradient(model, device)
                    regularizer.apply_projection(model, g_cur, device)
                optimizer.step()


# ══════════════════════════════════════════════════════════════════════════════
# Argument parsing
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="FedMMA multi-task federated learning on Waymo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core experiment
    p.add_argument("--baseline",      type=str, default="fedmma",
                   choices=["fedmma","fedavg","fedprox","local","sequential","multitask"])
    p.add_argument("--arriving-task", type=str, default="seg3d",
                   choices=TASKS, dest="arriving_task")

    # Data
    p.add_argument("--pt-dir",        type=str, required=True,  dest="pt_dir")
    p.add_argument("--num-clients",   type=int, default=20,     dest="num_clients")
    p.add_argument("--pan-seq-len",   type=int, default=5,      dest="pan_seq_len")
    p.add_argument("--eval-frac",     type=float, default=0.1,  dest="eval_frac")
    p.add_argument("--num-workers",   type=int, default=4,      dest="num_workers")

    # Training schedule
    p.add_argument("--phase1-rounds", type=int, default=100,    dest="phase1_rounds")
    p.add_argument("--phase2-rounds", type=int, default=50,     dest="phase2_rounds")
    p.add_argument("--local-epochs",  type=int, default=1,      dest="local_epochs")
    p.add_argument("--batch-size",    type=int, default=8,      dest="batch_size")
    p.add_argument("--frac",          type=float, default=1.0,
                   help="Fraction of clients sampled per round")
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--weight-decay",  type=float, default=1e-2, dest="weight_decay")
    p.add_argument("--prec",          type=str, default="amp",  choices=["fp32","amp"])
    p.add_argument("--eval-interval", type=int, default=10,     dest="eval_interval")

    # Phase behaviour
    p.add_argument(
        "--train-pretrained-heads",
        action="store_true", default=False, dest="train_pretrained_heads",
        help=(
            "Phase 2 trains ALL four task heads (multi-task fine-tune paradigm). "
            "Default: only the arriving head trains (continual-learning paradigm)."
        ),
    )
    p.add_argument(
        "--phase1-reg",
        type=str, default="none",
        choices=["none", "cagrad", "gradnorm"],
        dest="phase1_reg",
        help="Gradient-conflict mitigation for phase 1 multi-task training.",
    )
    p.add_argument(
        "--phase2-reg",
        type=str, default="none",
        choices=["none", "cagrad", "gradnorm", "ewc", "agem", "fedewc"],
        dest="phase2_reg",
        help=(
            "Phase 2 gradient-conflict or CL method (independent of --phase2-prox). "
            "MTL paradigm (--train-pretrained-heads): none | cagrad | gradnorm. "
            "CL paradigm (default): none | ewc | agem | fedewc."
        ),
    )
    p.add_argument(
        "--phase2-prox",
        action="store_true", default=False, dest="phase2_prox",
        help=(
            "Enable server-side Fisher-weighted proximal blend after each phase-2 "
            "aggregation.  Compatible with any --phase2-reg choice (none / cagrad / "
            "gradnorm in MTL paradigm; none / ewc / agem / fedewc in CL paradigm). "
            "Requires a federated baseline with cross-client aggregation "
            "(fedmma / fedavg / fedprox)."
        ),
    )

    # Regularisation hyper-parameters
    p.add_argument("--mu",              type=float, default=0.01,
                   help="FedProx / FedProxAnchor proximal coefficient.")
    p.add_argument("--ewc-lambda",      type=float, default=5000.0,  dest="ewc_lambda")
    p.add_argument("--ewc-max-batches", type=int,   default=50,      dest="ewc_max_batches",
                   help="Batches used to estimate Fisher for EWC/FedEWC.")
    p.add_argument("--agem-memory-size",type=int,   default=256,     dest="agem_memory_size")
    p.add_argument("--cagrad-alpha",    type=float, default=0.5,     dest="cagrad_alpha")
    p.add_argument("--gradnorm-alpha",  type=float, default=1.5,     dest="gradnorm_alpha")

    # Model architecture
    p.add_argument("--backbone",            type=str,   default="ViT-B/16")
    p.add_argument("--adapter-dim",         type=int,   default=64,  dest="adapter_dim")
    p.add_argument("--shared-adapter-dim",  type=int,   default=64,  dest="shared_adapter_dim")
    p.add_argument("--adapter-start",       type=int,   default=5,   dest="adapter_start")
    p.add_argument("--adapter-scale",       type=float, default=0.1, dest="adapter_scale")
    p.add_argument("--num-decoder-layers",  type=int,   default=2,   dest="num_decoder_layers")
    p.add_argument("--num-queries-3d",      type=int,   default=300, dest="num_queries_3d")
    p.add_argument("--num-queries-2d",      type=int,   default=100, dest="num_queries_2d")
    p.add_argument("--num-queries-pan",     type=int,   default=100, dest="num_queries_pan")
    p.add_argument("--no-task-adapter",     action="store_false", dest="use_task_adapter")
    p.set_defaults(use_task_adapter=True)

    # I/O
    p.add_argument("--output-dir", type=str, default="output/waymo", dest="output_dir")
    p.add_argument("--resume",     type=str, default=None)
    p.add_argument("--seed",       type=int, default=1)
    p.add_argument("--device",     type=str, default="cuda")

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    _validate_args(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(
        args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    )

    paradigm = "MTL fine-tune" if args.train_pretrained_heads else "continual-learning"
    print(f"Device       : {device}")
    print(f"Baseline     : {args.baseline}  |  Arriving task: {args.arriving_task}")
    print(f"Paradigm     : {paradigm}")
    prox_flag = " + server-prox" if args.phase2_prox else ""
    print(f"Phase 1      : {args.phase1_rounds} rounds  |  reg={args.phase1_reg}")
    print(f"Phase 2      : {args.phase2_rounds} rounds  |  reg={args.phase2_reg}{prox_flag}")

    run_name     = f"{args.baseline}_{args.arriving_task}_seed{args.seed}"
    args.output_dir = os.path.join(args.output_dir, run_name)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print("Building CustomCLIP …")
    model   = build_model(args).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_train:,}")

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["global_sd"], strict=False)
        print(f"Resumed from {args.resume}")

    if args.baseline in CENTRALIZED_BASELINES:
        metrics_log = run_centralised(args, model, device)
    else:
        _, _, metrics_log = run_federated(args, model, device)

    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics_log, f, indent=2)
    print(f"\nDone. Results → {args.output_dir}/metrics.json")


if __name__ == "__main__":
    main()
