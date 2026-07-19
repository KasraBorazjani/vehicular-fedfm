#!/usr/bin/env python3
"""
scripts/smoke_test_regularizers.py
------------------------------------
Smoke tests for models/regularizers.py:
  - EWC (consolidate + penalty)
  - FedEWC (aggregate_fishers)
  - AGEM (update_memory + project_gradient)
  - FedProxAnchor (set_global + penalty)

Uses a tiny MLP as a surrogate model. No CLIP or Waymo data required.

Run from repo root:
    python scripts/smoke_test_regularizers.py
"""

import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from models.regularizers import EWC, FedEWC, AGEM, FedProxAnchor

print("=" * 60)
print("Regularizers smoke test")
print("=" * 60)

DEVICE = torch.device("cpu")


# ── Tiny surrogate model ─────────────────────────────────────────────────────

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 8)
        self.fc2 = nn.Linear(8, 4)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


# ── Minimal DataLoader mock ───────────────────────────────────────────────────

class MockBatch:
    """A single batch that looks like a Waymo seg3d batch."""
    def __init__(self, B=2):
        self.B = B

    def __iter__(self):
        for _ in range(3):   # 3 "batches"
            yield {
                "task":      "seg3d",
                "image":     torch.randn(self.B, 3, 224, 224),
                "lidar":     torch.randn(self.B, 3, 224, 224),
                "targets":   torch.randint(0, 22, (self.B, 224, 224)),
            }


# ── EWC ──────────────────────────────────────────────────────────────────────

print("\n[1] EWC — consolidate + penalty ...")

# We can't call model.task_heads[task] on TinyModel, so monkey-patch a mock head.
class _MockHead(nn.Module):
    def compute_loss(self, output, targets):
        return {"loss": output.sum()}

model = TinyModel().to(DEVICE)
model.task_heads = {"seg3d": _MockHead()}
model.task_heads["seg3d"].requires_grad_(False)

ewc = EWC(model, lambda_=1000.0)

# Consolidate: compute Fisher from mock batches
class _MockLoader:
    def __iter__(self):
        for _ in range(3):
            yield {
                "task":      "seg3d",
                "image":     torch.randn(2, 3, 224, 224),
                "lidar":     torch.randn(2, 3, 224, 224),
                "targets":   torch.randint(0, 22, (2, 224, 224)),
            }
    def __call__(self, task, image, lidar=None):
        return image.view(image.shape[0], -1)[:, :16]

# Monkey-patch model.__call__ to avoid CLIP architecture
_orig_call = type(model).__call__

def _mock_forward(self, task, image, lidar=None):
    return self.fc2(torch.relu(self.fc1(image.view(image.shape[0], -1)[:, :16])))

type(model).__call__ = _mock_forward

ewc.consolidate(model, _MockLoader(), task="seg3d", device=DEVICE, max_batches=3)

assert len(ewc._fishers) > 0, "No Fisher stored"
assert len(ewc._means)   > 0, "No means stored"

# Penalty before parameter change should be 0
penalty_before = ewc.penalty(model)
assert penalty_before.item() >= 0.0

# Perturb parameters → penalty should increase
with torch.no_grad():
    for p in model.parameters():
        p.add_(torch.randn_like(p) * 5.0)

penalty_after = ewc.penalty(model)
assert penalty_after.item() > penalty_before.item(), (
    f"EWC penalty did not increase after perturbation: "
    f"{penalty_before.item():.4f} → {penalty_after.item():.4f}"
)
print(f"    EWC penalty: before={penalty_before.item():.4f}  after={penalty_after.item():.4f}  OK")

# state_dict round-trip
sd = ewc.state_dict()
ewc2 = EWC(model, lambda_=1000.0)
ewc2.load_state_dict(sd)
assert torch.allclose(ewc2.penalty(model), ewc.penalty(model))
print(f"    EWC state_dict round-trip  OK")

type(model).__call__ = _orig_call


# ── FedEWC aggregate_fishers ──────────────────────────────────────────────────

print("\n[2] FedEWC.aggregate_fishers ...")

# Make two FedEWC instances with fake Fishers
ewc_a = FedEWC(TinyModel(), lambda_=1000.0)
ewc_b = FedEWC(TinyModel(), lambda_=1000.0)
for n, p in TinyModel().named_parameters():
    ewc_a._fishers[n] = torch.ones_like(p) * 2.0
    ewc_b._fishers[n] = torch.ones_like(p) * 4.0
    ewc_a._means[n]   = torch.zeros_like(p)
    ewc_b._means[n]   = torch.zeros_like(p)

# Equal-weight aggregate → should be 3.0
agg_fishers = FedEWC.aggregate_fishers(
    [ewc_a, ewc_b], dataset_sizes=[100, 100]
)
for n, f in agg_fishers.items():
    assert torch.allclose(f, torch.ones_like(f) * 3.0), f
print(f"    FedEWC.aggregate_fishers (equal weights → 3.0)  OK")


# ── AGEM ─────────────────────────────────────────────────────────────────────

print("\n[3] AGEM — update_memory + project_gradient ...")

model3 = TinyModel().to(DEVICE)
agem   = AGEM(memory_size=32)

# update_memory with mock loader
class _MockDL:
    def __iter__(self):
        for _ in range(5):
            yield {
                "task":      "seg3d",
                "image":     torch.randn(2, 3, 224, 224),
                "lidar":     None,
                "targets":   torch.randint(0, 22, (2, 224, 224)),
            }

agem.update_memory(_MockDL(), arriving_task="seg3d", device=DEVICE, max_samples=10)
assert len(agem._memory) > 0, "Memory is empty after update"
assert len(agem._memory) <= agem.memory_size
print(f"    AGEM memory after update: {len(agem._memory)} entries  OK")

# project_gradient: just check it doesn't crash
type(model3).__call__ = _mock_forward
model3.task_heads = {"seg3d": _MockHead()}
agem.project_gradient(model3, device=DEVICE)
assert hasattr(agem, "_last_g_ref"), "g_ref not stored"
print(f"    AGEM.project_gradient ran without error  OK")

type(model3).__call__ = _orig_call


# ── FedProxAnchor ─────────────────────────────────────────────────────────────

print("\n[4] FedProxAnchor — set_global + penalty ...")

model4 = TinyModel().to(DEVICE)
anchor = FedProxAnchor(mu=0.1)

# No global set → penalty should be 0
p0 = anchor.penalty(model4)
assert p0.item() == 0.0, p0

# Set global = current params
anchor.set_global(model4)
p_same = anchor.penalty(model4)
assert abs(p_same.item()) < 1e-8, f"Expected ~0, got {p_same.item()}"

# Perturb → penalty should increase
with torch.no_grad():
    for p in model4.parameters():
        p.add_(torch.randn_like(p) * 2.0)

p_diff = anchor.penalty(model4)
assert p_diff.item() > 0.0, f"Expected >0, got {p_diff.item()}"
print(f"    FedProxAnchor penalty: same=0.0  after perturb={p_diff.item():.4f}  OK")

# Penalty contributes to backward
loss = anchor.penalty(model4)
loss.backward()
has_grad = any(p.grad is not None for p in model4.parameters())
assert has_grad, "FedProxAnchor penalty has no gradient"
print(f"    FedProxAnchor backward: gradient present  OK")


print("\n" + "=" * 60)
print("All regularizer smoke tests passed.")
print("=" * 60)
