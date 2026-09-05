"""Callbacks: encoder freezing, TensorBoard previews, and crash-safe exits."""
from __future__ import annotations

import signal
import time
from pathlib import Path

import lightning.pytorch as pl
import torch


class FreezeEncoder(pl.Callback):
    """Hold ``swinViT`` fixed for the first ``freeze_epochs``.

    finetuning.md Sec. 5: at n=38 the encoder is the part you cannot afford to
    retrain. Note the interaction that cost a whole run: freezing *and* a
    warm-up *and* a small base LR compose multiplicatively, and the first ten
    epochs then do nothing at all. Pick one or two of the three, not all.
    """

    def __init__(self, freeze_epochs: int = 3, attr: str = "swinViT") -> None:
        super().__init__()
        self.freeze_epochs = freeze_epochs
        self.attr = attr
        self._frozen = False

    def _encoder(self, pl_module):
        return getattr(pl_module.net, self.attr, None)

    def on_train_start(self, trainer, pl_module):
        if self.freeze_epochs <= 0 or trainer.current_epoch >= self.freeze_epochs:
            return
        enc = self._encoder(pl_module)
        if enc is None:
            return
        for p in enc.parameters():
            p.requires_grad_(False)
        self._frozen = True
        pl_module.print(f"[freeze] {self.attr} frozen until epoch {self.freeze_epochs}")

    def on_train_epoch_start(self, trainer, pl_module):
        if self._frozen and trainer.current_epoch >= self.freeze_epochs:
            enc = self._encoder(pl_module)
            if enc is not None:
                for p in enc.parameters():
                    p.requires_grad_(True)
            self._frozen = False
            pl_module.print(f"[freeze] {self.attr} unfrozen at epoch {trainer.current_epoch}")


class LogValidationSlices(pl.Callback):
    """Write an image / ground-truth / prediction strip to TensorBoard.

    Per-channel Dice tells you a number is small; it does not tell you whether
    the prediction is a fragmented dusting or a coherent branch that stops
    early -- and those want opposite fixes. One glance at the strip does.
    """

    def __init__(self, every_n_epochs: int = 5, max_channels: int = 6) -> None:
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.max_channels = max_channels
        self._batch = None

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, *_):
        if batch_idx == 0 and outputs is not None:
            self._batch = (outputs["logits"].detach(), outputs["labels"].detach())

    def on_validation_epoch_end(self, trainer, pl_module):
        if self._batch is None or trainer.sanity_checking:
            return
        if trainer.current_epoch % self.every_n_epochs:
            self._batch = None
            return
        logger = getattr(trainer, "logger", None)
        writer = getattr(logger, "experiment", None)
        if writer is None or not hasattr(writer, "add_image"):
            self._batch = None
            return

        logits, labels = self._batch
        self._batch = None
        probs = torch.sigmoid(logits[0].float().cpu())
        target = labels[0].float().cpu()
        # The axial slice with the most foreground across all channels: the
        # middle slice of a liver crop is often vessel-free.
        weight = target.sum(dim=(0, 1, 2))
        z = int(torch.argmax(weight)) if float(weight.sum()) > 0 else target.shape[-1] // 2

        rows = []
        for c in range(min(self.max_channels, probs.shape[0])):
            rows.append(torch.stack([target[c, :, :, z], probs[c, :, :, z]], dim=0))
        grid = torch.cat([torch.cat(list(r), dim=1) for r in rows], dim=0)
        writer.add_image("val/gt_vs_pred", grid.unsqueeze(0).clamp(0, 1),
                         global_step=trainer.global_step)


def _label(key: str) -> str:
    """``val/dice_portal_tree`` -> ``portal_tree``, ``train/loss_epoch`` ->
    ``train_loss`` -- so the epoch line reads at a glance rather than being a
    column of near-identical prefixes."""
    if key.startswith("val/dice_"):
        return key[len("val/dice_"):]
    return {"train/loss_epoch": "train_loss", "val/loss": "val_loss", "val/dice": "dice_mean"}.get(
        key, key.split("/")[-1])


class EpochLine(pl.Callback):
    """One line per validation epoch on stdout.

    The detached launcher turns the progress bar off -- a tqdm bar rewriting
    itself into a log file produces megabytes of escape codes and no readable
    history. This prints what the bar would have shown, once per epoch, so
    ``run.sh logs`` and ``run.sh status`` have something to report.
    """

    def __init__(self, keys: tuple[str, ...] = ()) -> None:
        super().__init__()
        self.keys = keys
        self._t0 = None

    def on_train_epoch_start(self, trainer, pl_module):
        self._t0 = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        # Not ``on_validation_epoch_end``: Lightning runs validation *inside*
        # the training epoch and calls callbacks before the LightningModule's
        # own hook, so at that point ``val/dice_*`` is still the previous
        # epoch's and ``train/loss_epoch`` does not exist yet. By
        # ``on_train_epoch_end`` both are current.
        if trainer.sanity_checking:
            return
        m = {k: float(v) for k, v in trainer.callback_metrics.items()
             if isinstance(v, (int, float, torch.Tensor))}
        keys = self.keys or [k for k in ("train/loss_epoch", "val/loss") if k in m] + \
            sorted(k for k in m if k.startswith("val/dice"))
        elapsed = f"{time.time() - self._t0:5.1f}s" if self._t0 else "    -"
        lr = trainer.optimizers[0].param_groups[-1]["lr"] if trainer.optimizers else float("nan")
        body = "  ".join(f"{_label(k)}={m[k]:.4f}" for k in keys if k in m)
        print(f"[epoch {trainer.current_epoch:4d}] {elapsed}  lr={lr:.2e}  {body}", flush=True)


class SaveOnSignal(pl.Callback):
    """Write ``interrupted.ckpt`` when the process is asked to stop.

    ``run.sh stop`` sends SIGTERM; Lightning's own handler ends the run at the
    next safe point but does not guarantee a checkpoint from mid-epoch. This
    writes one immediately, next to ``last.ckpt``, so a stop never costs more
    than the epoch in flight.
    """

    def __init__(self, ckpt_dir: str | Path) -> None:
        super().__init__()
        self.ckpt_dir = Path(ckpt_dir)
        self._installed = False

    def on_train_start(self, trainer, pl_module):
        if self._installed:
            return
        self._installed = True
        previous = {}

        def handler(signum, frame):
            try:
                self.ckpt_dir.mkdir(parents=True, exist_ok=True)
                path = self.ckpt_dir / "interrupted.ckpt"
                trainer.save_checkpoint(str(path))
                print(f"\n[signal {signum}] wrote {path}", flush=True)
            except Exception as exc:  # never let the handler mask the shutdown
                print(f"\n[signal {signum}] could not checkpoint: {exc}", flush=True)
            prev = previous.get(signum)
            if callable(prev):
                prev(signum, frame)

        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                previous[sig] = signal.getsignal(sig)
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass
