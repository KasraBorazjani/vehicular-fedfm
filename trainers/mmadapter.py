"""
trainers/mmadapter.py
---------------------
MultiModalAdapter: thin Dassl TrainerX wrapper around CustomCLIP.

All model architecture lives in models/. This file only handles:
  - CLIP weight loading
  - Optimizer / scheduler construction
  - Per-batch forward + loss computation
  - Checkpoint save / load
"""

import os.path as osp

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from Dassl.dassl.engine.trainer import TrainerX
from Dassl.dassl.utils import load_pretrained_weights, load_checkpoint
from Dassl.dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from models.custom_clip import CustomCLIP


def load_clip_to_cpu(cfg) -> nn.Module:
    """Download (or reuse cached) CLIP weights and build the MMA-modified model."""
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url           = clip._MODELS[backbone_name]
    model_path    = clip._download(url)
    try:
        model      = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    return clip.build_model_MMA(state_dict or model.state_dict())


class MultiModalAdapter(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.MMADAPTER.PREC in ("fp16", "fp32", "amp")

    def build_model(self):
        cfg = self.cfg

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.MMADAPTER.PREC in ("fp32", "amp"):
            clip_model.float()

        print("Building CustomCLIP")
        self.model = CustomCLIP(
            clip_model              = clip_model,
            arriving_task           = cfg.TRAINER.MMADAPTER.ARRIVING_TASK,
            adapter_dim             = cfg.TRAINER.MMADAPTER.ADAPTER_DIM,
            shared_adapter_dim      = cfg.TRAINER.MMADAPTER.SHARED_ADAPTER_DIM,
            adapter_start           = cfg.TRAINER.MMADAPTER.ADAPTER_START,
            adapter_scale           = cfg.TRAINER.MMADAPTER.ADAPTER_SCALE,
            num_decoder_layers      = cfg.MODEL.NUM_DECODER_LAYERS,
            num_queries_3d          = cfg.MODEL.NUM_QUERIES_3D,
            num_queries_2d          = cfg.MODEL.NUM_QUERIES_2D,
            use_task_adapter        = cfg.TRAINER.MMADAPTER.USE_TASK_ADAPTER,
            freeze_pretrained_heads = cfg.TRAINER.MMADAPTER.FREEZE_PRETRAINED_HEADS,
        )

        trainable = [(n, p.numel()) for n, p in self.model.named_parameters()
                     if p.requires_grad]
        total = sum(n for _, n in trainable)
        print(f"Trainable parameters ({total:,}):")
        for name, n in trainable:
            print(f"  {name}  ({n:,})")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("custom_clip", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.MMADAPTER.PREC == "amp" else None

    # ── Training step ─────────────────────────────────────────────────────────

    def forward_backward(self, batch_idx, batch, **kwargs):
        task, image, lidar, targets = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.MMADAPTER.PREC

        if prec == "amp":
            with autocast():
                output = self.model(task, image, lidar)
                losses = self._compute_loss(task, output, targets)
            self.optim.zero_grad()
            self.scaler.scale(losses["loss"]).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(task, image, lidar)
            losses = self._compute_loss(task, output, targets)
            self.model_backward_and_update(losses["loss"])

        if (batch_idx + 1) == self.num_batches:
            self.update_lr()

        return {k: v.item() for k, v in losses.items()}

    def _compute_loss(self, task, output, targets):
        head = self.model.task_heads[task]
        if task in ("seg3d", "det3d", "pan2d"):
            # Heads unpack output internally
            return head.compute_loss(output, targets)
        else:   # det2d
            cls_logits, box_preds = output
            return head.compute_loss(cls_logits, box_preds, targets)

    def parse_batch_train(self, batch):
        task  = batch["task"]
        image = batch["image"].to(self.device)

        lidar = batch.get("lidar")
        if lidar is not None:
            lidar = lidar.to(self.device)

        targets = batch["targets"]
        return task, image, lidar, targets

    # ── Checkpoint ────────────────────────────────────────────────────────────

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note: load_model() skipped — no pretrained model given")
            return

        names      = self.get_model_names()
        model_file = "model-best.pth.tar" if epoch is None else f"model.pth.tar-{epoch}"

        for name in names:
            model_path = osp.join(directory, name, model_file)
            if not osp.exists(model_path):
                raise FileNotFoundError(f'Model not found at "{model_path}"')
            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch      = checkpoint["epoch"]
            print(f"Loading weights to {name} from '{model_path}' (epoch={epoch})")
            self._models[name].load_state_dict(state_dict, strict=False)
