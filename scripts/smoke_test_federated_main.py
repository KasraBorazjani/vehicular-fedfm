#!/usr/bin/env python3
"""
scripts/smoke_test_federated_main.py
--------------------------------------
Dry-run smoke test for federated_main.py.

Tests key functions (key partitioning, local_train one step, aggregation,
evaluation) using:
  - Mock .pt segment files (tiny images, no real Waymo data)
  - Mock CustomCLIP (_MockCLIP, same as in smoke_test_heads_and_forward.py)

Does NOT download CLIP weights; does NOT run full rounds.

Run from repo root:
    python scripts/smoke_test_federated_main.py
"""

import sys, os, tempfile, shutil, types, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

print("=" * 60)
print("federated_main dry-run smoke test")
print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# Mock .pt files
# ══════════════════════════════════════════════════════════════════════════════

def make_mock_pt_dir(n_files=4, frames_per_file=10, pan_seq_len=5):
    tmp = tempfile.mkdtemp(prefix="fedmma_fed_smoke_")
    for i in range(n_files):
        frames = []
        for _ in range(frames_per_file):
            N = 3
            frames.append({
                "image":          torch.zeros(3, 224, 224),
                "lidar":          torch.zeros(3, 224, 224),
                "seg3d_label":    torch.randint(0, 22, (224, 224), dtype=torch.int16),
                "det3d_boxes":    torch.rand(N, 8),
                "pan2d_label":    torch.randint(0, 28, (224, 224), dtype=torch.int32),
                "pan2d_instance": torch.zeros((224, 224), dtype=torch.int32),
                "det2d_boxes":    torch.rand(N, 5),
            })
        torch.save({"frames": frames}, os.path.join(tmp, f"seg_{i:04d}.pt"))
    return tmp


# ══════════════════════════════════════════════════════════════════════════════
# Mock CustomCLIP (avoids CLIP download)
# ══════════════════════════════════════════════════════════════════════════════

from models.task_heads import SegHead3D, DetHead3D, PanHead2D, DetHead2D, build_task_heads
from models.adapters import SharedTaskAdapter

B, T, D = 2, 5, 512

class _MockVisual(nn.Module):
    def forward(self, args):
        x, _ = args
        B_   = x.shape[0]
        return torch.randn(B_, 196, D, device=x.device, dtype=x.dtype)

class _MockTransformer(nn.Module):
    def forward(self, args):
        return args[0] if isinstance(args, (list, tuple)) else args

class _MockClipModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual               = _MockVisual()
        self.dtype                = torch.float32
        self.transformer          = _MockTransformer()
        self.positional_embedding = nn.Parameter(torch.randn(77, 768))
        self.ln_final             = nn.LayerNorm(768)
        self.text_projection      = nn.Parameter(torch.randn(768, D))
        self.token_embedding      = nn.Embedding(49408, 768)
    def requires_grad_(self, flag=True):
        for p in self.parameters():
            p.requires_grad_(flag)
        return self

# Patch CustomCLIP to skip real CLIP architecture dependencies
from models import custom_clip as _cc_mod
from models.custom_clip import CustomCLIP

_OrigBuildText = CustomCLIP._build_text_prompts

def _mock_build_text(self, clip_model):
    n_classes = 22 + 4 + 28 + 3  # 57 total
    self.register_buffer("token_embedding",   torch.randn(n_classes, 77, 768))
    self.register_buffer("tokenized_prompts", torch.zeros(n_classes, 77, dtype=torch.long))

def _mock_encode_text(self):
    # [57, 512]
    return torch.randn(57, D, device=next(self.parameters()).device)

CustomCLIP._build_text_prompts = _mock_build_text
CustomCLIP._encode_text        = _mock_encode_text


# ══════════════════════════════════════════════════════════════════════════════
# Build a tiny CustomCLIP with the mock backbone
# ══════════════════════════════════════════════════════════════════════════════

print("\n[1] Building mock CustomCLIP (seg3d arriving task) ...")
DEVICE = torch.device("cpu")

model = CustomCLIP(
    clip_model              = _MockClipModel(),
    arriving_task           = "seg3d",
    adapter_dim             = 8,
    shared_adapter_dim      = 8,
    adapter_start           = 11,   # last layer only, keeps model tiny
    adapter_scale           = 0.1,
    num_decoder_layers      = 1,
    num_queries_3d          = 10,
    num_queries_2d          = 10,
    num_queries_pan         = 10,
    use_task_adapter        = True,
    freeze_pretrained_heads = True,
).to(DEVICE)

n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"    trainable params: {n_train:,}  OK")


# ══════════════════════════════════════════════════════════════════════════════
# Key partitioning
# ══════════════════════════════════════════════════════════════════════════════

print("\n[2] Key partitioning ...")
from federated_main import _global_keys, _local_keys, _trainable_keys

g_keys = _global_keys(model)
l_keys = _local_keys(model, "seg3d")
t_keys = _trainable_keys(model)

assert len(g_keys) > 0, "No global keys"
assert len(l_keys) > 0, "No local keys"
assert len(t_keys) > 0, "No trainable keys"
# Global and local should be disjoint
assert set(g_keys).isdisjoint(set(l_keys)), "Global/local overlap"
print(f"    global={len(g_keys)}  local={len(l_keys)}  trainable={len(t_keys)}  OK")


# ══════════════════════════════════════════════════════════════════════════════
# compute_loss
# ══════════════════════════════════════════════════════════════════════════════

print("\n[3] compute_loss (seg3d) ...")
from federated_main import compute_loss as _compute_loss

seg_batch = {
    "task":      "seg3d",
    "image":     torch.randn(B, 3, 224, 224),
    "lidar":     torch.randn(B, 3, 224, 224),
    "targets":   torch.randint(0, 22, (B, 224, 224)),
}
losses = _compute_loss(model, seg_batch, "seg3d", DEVICE)
assert "loss" in losses
assert not losses["loss"].isnan(), "NaN loss"
print(f"    seg3d loss={losses['loss'].item():.4f}  OK")


# ══════════════════════════════════════════════════════════════════════════════
# local_train — one batch, one epoch
# ══════════════════════════════════════════════════════════════════════════════

print("\n[4] local_train (1 batch, 1 epoch) ...")
from federated_main import local_train

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad], lr=1e-4
)

class _OneLoader:
    """DataLoader stub that yields exactly one batch."""
    def __iter__(self):
        yield {
            "task":      "seg3d",
            "image":     torch.randn(B, 3, 224, 224),
            "lidar":     torch.randn(B, 3, 224, 224),
            "targets":   torch.randint(0, 22, (B, 224, 224)),
        }

final_sd = local_train(
    model        = model,
    loader       = _OneLoader(),
    optimizer    = optimizer,
    scaler       = None,
    device       = DEVICE,
    arriving_task= "seg3d",
    local_epochs = 1,
)
assert isinstance(final_sd, dict)
assert all(v.device.type == "cpu" for v in final_sd.values())
print(f"    local_train returned state_dict ({len(final_sd)} keys)  OK")


# ══════════════════════════════════════════════════════════════════════════════
# fedavg_aggregate — simulate 2 client updates
# ══════════════════════════════════════════════════════════════════════════════

print("\n[5] fedavg_aggregate (2 clients) ...")
import copy
from utils.fed_utils import fedavg_aggregate, fedmma_aggregate

sd_a = copy.deepcopy(final_sd)
sd_b = copy.deepcopy(final_sd)
# Perturb sd_b
for k in sd_b:
    if sd_b[k].is_floating_point():
        sd_b[k] = sd_b[k] + 1.0

agg_all = fedavg_aggregate([sd_a, sd_b], weights=[1.0, 1.0], keys=t_keys)
# The aggregated value should be between sd_a and sd_b
for k in t_keys:
    if k in agg_all:
        assert torch.allclose(agg_all[k], (sd_a[k].float() + sd_b[k].float()) / 2, atol=1e-5)
        break   # spot-check one key
print(f"    fedavg_aggregate ({len(t_keys)} trainable keys)  OK")

agg_global = fedmma_aggregate([sd_a, sd_b], global_keys=g_keys, weights=[1.0, 1.0])
assert len(agg_global) == len(g_keys)
print(f"    fedmma_aggregate ({len(g_keys)} global keys)  OK")


# ══════════════════════════════════════════════════════════════════════════════
# FedProxAnchor integration with local_train
# ══════════════════════════════════════════════════════════════════════════════

print("\n[6] FedProxAnchor + local_train ...")
from models.regularizers import FedProxAnchor

anchor = FedProxAnchor(mu=0.01)
anchor.set_global(model)

optimizer2 = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad], lr=1e-4
)
final_sd2 = local_train(
    model, _OneLoader(), optimizer2, None, DEVICE, "seg3d", 1,
    regularizer=anchor,
)
assert isinstance(final_sd2, dict)
print(f"    local_train + FedProxAnchor: {len(final_sd2)} keys returned  OK")


# ══════════════════════════════════════════════════════════════════════════════
# Data manager + full one-round federated loop
# ══════════════════════════════════════════════════════════════════════════════

print("\n[7] One federated round (fedmma, seg3d, mock data) ...")

pt_dir = make_mock_pt_dir(n_files=4, frames_per_file=10, pan_seq_len=5)
try:
    from utils.waymo_data_manager import WaymoDataManager

    mgr = WaymoDataManager(
        pt_dir        = pt_dir,
        num_clients   = 2,
        arriving_task = "seg3d",
        batch_size    = 2,
        pan_seq_len   = 5,
        num_workers   = 0,
        seed          = 1,
        eval_frac     = 0.25,
    )

    import random
    global_sd  = {k: v.cpu() for k, v in model.state_dict().items()}
    local_sds  = [copy.deepcopy(global_sd) for _ in range(2)]
    local_upd: list = []
    w_sampled:  list = []

    for c_id in [0, 1]:
        init_sd = copy.deepcopy(global_sd)
        init_sd.update({k: local_sds[c_id][k] for k in l_keys if k in local_sds[c_id]})
        model.load_state_dict(init_sd, strict=False)

        opt   = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        loader = mgr.get_loader(c_id, "seg3d")

        sd_out = local_train(model, loader, opt, None, DEVICE, "seg3d", 1)
        local_upd.append(sd_out)
        w_sampled.append(float(mgr.dataset_sizes(c_id)["seg3d"] + 1))

    new_global = fedmma_aggregate(local_upd, g_keys, w_sampled)
    global_sd.update(new_global)
    print(f"    FedMMA one round: {len(new_global)} global keys updated  OK")

finally:
    shutil.rmtree(pt_dir, ignore_errors=True)


# ══════════════════════════════════════════════════════════════════════════════
# save_checkpoint / load_latest_checkpoint
# ══════════════════════════════════════════════════════════════════════════════

print("\n[8] Checkpointing ...")
from federated_main import save_checkpoint, load_latest_checkpoint

ckpt_dir = tempfile.mkdtemp(prefix="fedmma_ckpt_")
try:
    save_checkpoint(ckpt_dir, round_idx=5, phase=1, global_sd=global_sd,
                    local_sds=None, metrics={"miou": 0.42})
    loaded = load_latest_checkpoint(ckpt_dir)
    assert loaded is not None
    assert loaded["round"]   == 5
    assert loaded["phase"]   == 1
    assert abs(loaded["metrics"]["miou"] - 0.42) < 1e-6
    print(f"    save + load checkpoint  OK")
finally:
    shutil.rmtree(ckpt_dir, ignore_errors=True)


# ══════════════════════════════════════════════════════════════════════════════

# Restore patched methods (good practice when running in isolation)
CustomCLIP._build_text_prompts = _OrigBuildText

print("\n" + "=" * 60)
print("All federated_main smoke tests passed.")
print("=" * 60)
