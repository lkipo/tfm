"""SwinUNETR fine-tuned from the BTCV checkpoint onto the cornerstone classes.

The architecture change is one layer. ``data/nn_model_weights/swin_unetr-epoch=
483-val_loss=0.07.ckpt`` is a MONAI SwinUNETR with ``feature_size=36`` and a
14-class BTCV head, stored by a LightningModule that named it ``_model.*``.
Stripping that prefix, 157 of its 159 tensors load into a ``feature_size=36``
SwinUNETR verbatim; the two that do not are ``out.conv.conv.{weight,bias}`` --
the head, which becomes ``len(channels)`` sigmoid outputs instead of 14 softmax
ones. Everything else -- the Swin encoder and the whole UNETR decoder -- is
transferred.

Two details of the new head matter more than they look:

* **It is sigmoid, not softmax.** The channels overlap by construction (see
  ``datamodule.BuildChannelsd``).
* **Its bias is initialised to each channel's log-odds prior.** At 0.03-0.3%
  prevalence a zero-initialised head starts by predicting p=0.5 everywhere,
  and the fastest way down the loss surface is to push every logit to -inf --
  the all-background collapse that ended the runs in ``training_diary.md``.
  Starting the bias at ``log(p/(1-p))`` puts the head at the base rate on step
  one, so the gradient carries shape information from the beginning. This is
  the focal-loss prior trick (Lin et al. 2017, Sec. 3.3), and it costs one
  line.
"""
from __future__ import annotations

import math
from typing import Any, Optional, Sequence

import lightning.pytorch as pl
import torch
from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR
from torch import nn

from losses import MultiLabelLoss, _broadcast_mask


class CornerstoneSegModule(pl.LightningModule):
    def __init__(
        self,
        channel_names: Sequence[str],
        thin_channels: Sequence[int] = (),
        in_channels: int = 1,
        feature_size: int = 36,
        roi_size: Sequence[int] = (96, 96, 96),
        sw_batch_size: int = 4,
        sw_overlap: float = 0.5,
        use_checkpoint: bool = True,
        pretrained_weights: Optional[str] = None,
        prevalence: Optional[Sequence[float]] = None,
        pos_weight: Optional[Sequence[float]] = None,
        pos_weight_cap: float = 50.0,
        init_bias_prior: bool = True,
        lambda_dice: float = 1.0,
        lambda_bce: float = 1.0,
        lambda_cldice: float = 0.5,
        cldice_iterations: int = 10,
        use_tversky: bool = False,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-2,
        encoder_lr_scale: float = 0.3,
        warmup_epochs: int = 2,
        max_epochs: int = 300,
        scheduler: str = "cosine",
        threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.channel_names = list(channel_names)
        n = len(self.channel_names)

        self.net = SwinUNETR(
            img_size=tuple(roi_size),
            in_channels=in_channels,
            out_channels=n,
            feature_size=feature_size,
            num_heads=(3, 6, 12, 24),
            use_checkpoint=use_checkpoint,
        )
        if pretrained_weights:
            self.load_pretrained(pretrained_weights)

        prevalence = list(prevalence) if prevalence else [1e-3] * n
        if init_bias_prior:
            self._init_head_bias(prevalence)

        if pos_weight is None:
            # Inverse prevalence, capped: the uncapped value reaches ~3000 for
            # the thinnest channels, which swamps the Dice term and destabilises
            # mixed precision.
            pos_weight = [min(pos_weight_cap, (1.0 - p) / max(p, 1e-6)) for p in prevalence]
        self.hparams.pos_weight = [float(w) for w in pos_weight]

        self.loss_fn = MultiLabelLoss(
            cldice_channels=list(thin_channels),
            pos_weight=self.hparams.pos_weight,
            lambda_dice=lambda_dice,
            lambda_bce=lambda_bce,
            lambda_cldice=lambda_cldice,
            cldice_iterations=cldice_iterations,
            use_tversky=use_tversky,
            tversky_alpha=tversky_alpha,
            tversky_beta=tversky_beta,
        )

        # Running per-channel Dice over the validation epoch. Accumulating
        # intersection and denominator separately (rather than averaging
        # per-case Dice) keeps a subject whose channel is a few hundred voxels
        # from dominating the average through a noisy per-case score.
        for buf in ("_inter", "_denom", "_seen"):
            self.register_buffer(buf, torch.zeros(n), persistent=False)

    # ------------------------------------------------------------------
    def load_pretrained(self, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        for prefix in ("_model.", "net.", "module."):
            if any(k.startswith(prefix) for k in state):
                state = {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state.items()}
                break
        own = self.net.state_dict()
        compatible = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
        skipped = sorted(set(state) - set(compatible))
        self.net.load_state_dict(compatible, strict=False)
        print(
            f"[pretrained] {len(compatible)}/{len(own)} tensors from {path}; "
            f"skipped {len(skipped)}: {skipped}"
        )
        if len(compatible) < 0.9 * len(own):
            raise RuntimeError(
                f"only {len(compatible)}/{len(own)} tensors matched -- feature_size or "
                f"architecture disagrees with the checkpoint (expected feature_size=36)."
            )

    def _init_head_bias(self, prevalence: Sequence[float]) -> None:
        head = self.net.out.conv.conv
        with torch.no_grad():
            nn.init.normal_(head.weight, std=1e-3)
            for i, p in enumerate(prevalence):
                p = min(max(float(p), 1e-6), 1 - 1e-6)
                head.bias[i] = math.log(p / (1.0 - p))
        print("[head] bias prior: " + ", ".join(
            f"{n}={b:.2f}" for n, b in zip(self.channel_names, head.bias.tolist())))

    # ------------------------------------------------------------------
    def forward(self, x):
        return self.net(x)

    def _mask(self, batch, logits):
        return _broadcast_mask(batch.get("channel_mask"), logits)

    @staticmethod
    def _plain(t: torch.Tensor) -> torch.Tensor:
        """Drop MONAI's ``MetaTensor`` wrapper.

        Anything derived from a MetaTensor stays one, including the scalars
        handed to ``self.log``. ``ModelCheckpoint`` then stores a MetaTensor as
        ``best_model_score`` inside the checkpoint, and reloading it fails
        under torch>=2.6's ``weights_only=True`` default -- which is exactly
        when you need the checkpoint most: on resume, and on ``trainer.test
        (ckpt_path="best")``.
        """
        return t.as_tensor() if hasattr(t, "as_tensor") else t

    def training_step(self, batch, batch_idx):
        images, labels = self._plain(batch["image"]), self._plain(batch["label"])
        logits = self._plain(self.net(images))
        loss = self.loss_fn(logits, labels, channel_mask=self._mask(batch, logits))
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True,
                 batch_size=images.shape[0])
        return loss

    def _infer(self, images):
        return sliding_window_inference(
            inputs=images,
            roi_size=tuple(self.hparams.roi_size),
            sw_batch_size=self.hparams.sw_batch_size,
            predictor=self.net,
            overlap=self.hparams.sw_overlap,
            mode="gaussian",
        )

    def validation_step(self, batch, batch_idx):
        return self._shared_eval(batch, prefix="val")

    def _shared_eval(self, batch, prefix: str):
        images, labels = self._plain(batch["image"]), self._plain(batch["label"])
        logits = self._plain(self._infer(images))
        mask = self._mask(batch, logits)
        loss = self.loss_fn(logits, labels, channel_mask=mask)
        self.log(f"{prefix}/loss", loss, on_epoch=True, prog_bar=True, batch_size=images.shape[0])

        preds = (torch.sigmoid(logits) > self.hparams.threshold).float()
        dims = (0, 2, 3, 4)
        per_batch = (mask.sum(dim=0) > 0).float()
        self._inter += (preds * labels).sum(dim=dims) * per_batch
        self._denom += (preds.sum(dim=dims) + labels.sum(dim=dims)) * per_batch
        self._seen += per_batch
        return {"logits": logits, "labels": labels}

    def _epoch_end(self, prefix: str):
        dice = (2.0 * self._inter) / self._denom.clamp(min=1e-6)
        dice = torch.where(self._seen > 0, dice, torch.full_like(dice, float("nan")))
        logs = {f"{prefix}/dice_{n}": dice[i] for i, n in enumerate(self.channel_names)}
        logs[f"{prefix}/dice"] = torch.nanmean(dice)
        thin = list(self.hparams.thin_channels)
        if thin:
            # The metric to actually watch. A mean over channels is dominated
            # by ``liver`` -- 8% of the cropped volume against 0.1% for the
            # portal tree -- and hid a vessel Dice of 0.002 behind a headline
            # 0.06 in every previous run (finetuning.md Sec. 3).
            logs[f"{prefix}/dice_thin"] = torch.nanmean(dice[thin])
        bulk = [i for i in range(len(self.channel_names)) if i not in set(thin)]
        if bulk:
            logs[f"{prefix}/dice_bulk"] = torch.nanmean(dice[bulk])
        self.log_dict(logs, prog_bar=True, sync_dist=True)
        self._inter.zero_(); self._denom.zero_(); self._seen.zero_()

    def on_validation_epoch_end(self):
        self._epoch_end("val")

    def test_step(self, batch, batch_idx):
        return self._shared_eval(batch, prefix="test")

    def on_test_epoch_end(self):
        self._epoch_end("test")

    def predict_step(self, batch, batch_idx):
        logits = self._plain(self._infer(self._plain(batch["image"])))
        return {
            "probs": torch.sigmoid(logits),
            "case": batch.get("case"),
        }

    # ------------------------------------------------------------------
    def configure_optimizers(self) -> dict[str, Any]:
        encoder = getattr(self.net, "swinViT", None)
        if encoder is not None and self.hparams.encoder_lr_scale != 1.0:
            enc_ids = {id(p) for p in encoder.parameters()}
            groups = [
                {"params": [p for p in encoder.parameters()],
                 "lr": self.hparams.learning_rate * self.hparams.encoder_lr_scale},
                {"params": [p for p in self.parameters() if id(p) not in enc_ids],
                 "lr": self.hparams.learning_rate},
            ]
        else:
            groups = self.parameters()
        opt = torch.optim.AdamW(groups, lr=self.hparams.learning_rate,
                                weight_decay=self.hparams.weight_decay)
        if self.hparams.scheduler == "none":
            return {"optimizer": opt}
        if self.hparams.scheduler != "cosine":
            raise ValueError(f"unknown scheduler {self.hparams.scheduler!r}")

        # A warm-up stacked on top of a frozen encoder and a small base LR is
        # how the first run spent ten epochs at an effective 6e-6 and learned
        # nothing (``training_diary.md``, "LR trap"). Keep warmup short and
        # check the logged lr against the intended one on epoch 1.
        warmup = int(self.hparams.warmup_epochs)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, self.hparams.max_epochs - warmup))
        if warmup > 0:
            sched = torch.optim.lr_scheduler.SequentialLR(
                opt,
                schedulers=[
                    torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=warmup),
                    cosine,
                ],
                milestones=[warmup],
            )
        else:
            sched = cosine
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}
