#!/usr/bin/env python3
"""Fine-tune SwinUNETR on the cornerstone masks.

    python cornerstone_finetune/train.py --config cornerstone_finetune/configs/core6.yaml
    python cornerstone_finetune/train.py --config .../core6.yaml --set trainer.max_epochs=50
    python cornerstone_finetune/train.py --config .../core6.yaml --smoke

Restart-safety is the point of the ``--resume`` default: the run directory is
derived from the config's ``run_name``, and if ``checkpoints/last.ckpt`` is
there the trainer picks it up -- optimiser, scheduler, epoch and all. Re-running
the same command after a crash, an OOM, a reboot or a dropped SSH session
continues rather than restarts. ``run.sh start`` relies on exactly this.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import timedelta
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = HERE.parent


def deep_update(base: dict, key: str, value):
    node = base
    parts = key.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


def parse_scalar(text: str):
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="dotted-path override, e.g. --set model.learning_rate=1e-4")
    ap.add_argument("--resume", choices=["auto", "never", "interrupted"], default="auto",
                    help="auto: continue from checkpoints/last.ckpt if present (default)")
    ap.add_argument("--smoke", action="store_true",
                    help="two short epochs on two subjects: proves the wiring, not the model")
    ap.add_argument("--print-config", action="store_true", help="resolve config, print, exit")
    return ap


def main() -> int:
    args = build_argparser().parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    for override in args.set:
        if "=" not in override:
            raise SystemExit(f"--set expects KEY=VALUE, got {override!r}")
        key, value = override.split("=", 1)
        deep_update(cfg, key.strip(), parse_scalar(value.strip()))

    if args.smoke:
        cfg.setdefault("trainer", {}).update(
            {"max_epochs": 2, "limit_train_batches": 3, "limit_val_batches": 1,
             "num_sanity_val_steps": 0, "log_every_n_steps": 1})
        cfg.setdefault("data", {}).update({"num_workers": 0, "cache_rate_train": 0.0,
                                           "cache_rate_val": 0.0})
        cfg["run_name"] = cfg.get("run_name", "run") + "_smoke"

    if args.print_config:
        print(yaml.safe_dump(cfg, sort_keys=False))
        return 0

    # Imports are deferred so --print-config and --help stay instant and do not
    # need a GPU or a 20 s torch import.
    import lightning.pytorch as pl
    import torch
    from lightning.pytorch.callbacks import (
        EarlyStopping, LearningRateMonitor, ModelCheckpoint, ModelSummary)
    from lightning.pytorch.loggers import TensorBoardLogger

    from callbacks import EpochLine, FreezeEncoder, LogValidationSlices, SaveOnSignal
    from datamodule import CornerstoneDataModule
    from labels import get_channels, thin_channel_indices
    from module import CornerstoneSegModule

    # torch>=2.6 loads checkpoints with weights_only=True. Metrics are kept
    # free of MONAI MetaTensors (see module._plain), but allowlist the class
    # anyway so checkpoints written before that fix still resume.
    try:
        from monai.data import MetaTensor

        torch.serialization.add_safe_globals([MetaTensor])
    except Exception:  # pragma: no cover - older torch has no such API
        pass

    seed = int(cfg.get("seed", 42))
    pl.seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision(cfg.get("matmul_precision", "high"))

    run_root = Path(cfg.get("run_root", REPO / "runs" / "cornerstone"))
    run_dir = run_root / cfg.get("run_name", "run")
    ckpt_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = dict(cfg.get("data", {}))
    data_cfg.setdefault("prepared_root", str(REPO / "data" / "cornerstone_prepared"))
    data_cfg.setdefault("seed", seed)
    dm = CornerstoneDataModule(**data_cfg)
    if args.smoke:
        dm.hparams.samples_per_volume = 2
    print(dm.summary(), flush=True)

    channels = get_channels(dm.hparams.channel_preset)
    prevalence = dm.class_prevalence()
    print("[data] mean per-channel prevalence in train split: " + ", ".join(
        f"{n}={p:.5f}" for n, p in zip(dm.channel_names, prevalence)), flush=True)

    model_cfg = dict(cfg.get("model", {}))
    model_cfg.setdefault("roi_size", data_cfg.get("roi_size", (96, 96, 96)))
    model_cfg.setdefault("max_epochs", cfg.get("trainer", {}).get("max_epochs", 300))
    weights = model_cfg.get("pretrained_weights")
    if weights and not Path(weights).is_absolute():
        model_cfg["pretrained_weights"] = str((REPO / weights).resolve())
    model = CornerstoneSegModule(
        channel_names=dm.channel_names,
        thin_channels=thin_channel_indices(channels),
        prevalence=prevalence,
        **model_cfg,
    )
    print(f"[model] pos_weight = {[round(w, 1) for w in model.hparams.pos_weight]}", flush=True)

    monitor = cfg.get("monitor", "val/dice_thin")
    mode = cfg.get("monitor_mode", "max")
    callbacks = [
        ModelCheckpoint(
            dirpath=str(ckpt_dir), monitor=monitor, mode=mode,
            save_top_k=int(cfg.get("save_top_k", 3)), save_last=True,
            filename="epoch{epoch:03d}-" + monitor.replace("/", "_") + "{" + monitor + ":.4f}",
            auto_insert_metric_name=False,
        ),
        # A second, unconditional checkpoint on a wall-clock timer: the metric
        # checkpoint above only fires when the metric improves, so a run that
        # plateaus for hours and then dies would otherwise lose all of it.
        ModelCheckpoint(
            dirpath=str(ckpt_dir), filename="periodic",
            train_time_interval=timedelta(minutes=int(cfg.get("checkpoint_every_minutes", 30))),
            save_top_k=1, monitor=None,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        ModelSummary(max_depth=1),
        EpochLine(),
        SaveOnSignal(ckpt_dir),
    ]
    freeze_epochs = int(cfg.get("freeze_encoder_epochs", 0))
    if freeze_epochs:
        callbacks.append(FreezeEncoder(freeze_epochs=freeze_epochs))
    if cfg.get("log_slices_every_n_epochs", 5):
        callbacks.append(LogValidationSlices(
            every_n_epochs=int(cfg["log_slices_every_n_epochs"]),
            max_channels=len(dm.channel_names)))
    patience = int(cfg.get("early_stopping_patience", 0))
    if patience:
        callbacks.append(EarlyStopping(monitor=monitor, mode=mode, patience=patience,
                                       check_finite=False))

    logger = TensorBoardLogger(save_dir=str(run_dir), name="tb", default_hp_metric=False)

    trainer_cfg = {
        "accelerator": "gpu",
        "devices": 1,
        "precision": "bf16-mixed",
        "max_epochs": 300,
        "gradient_clip_val": 1.0,
        "log_every_n_steps": 5,
        "check_val_every_n_epoch": 1,
        "deterministic": "warn",
        "enable_progress_bar": bool(cfg.get("progress_bar", not os.environ.get("CORNERSTONE_DETACHED"))),
        **cfg.get("trainer", {}),
    }
    trainer = pl.Trainer(default_root_dir=str(run_dir), logger=logger,
                         callbacks=callbacks, **trainer_cfg)

    ckpt_path = None
    if args.resume == "auto" and (ckpt_dir / "last.ckpt").exists():
        ckpt_path = str(ckpt_dir / "last.ckpt")
    elif args.resume == "interrupted" and (ckpt_dir / "interrupted.ckpt").exists():
        ckpt_path = str(ckpt_dir / "interrupted.ckpt")
    print(f"[run] dir={run_dir}\n[run] resume={ckpt_path or 'from scratch'}\n"
          f"[run] tensorboard --logdir {run_dir / 'tb'}", flush=True)

    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (run_dir / "split.json").write_text(json.dumps(
        {name: [r["case"] for r in split]
         for name, split in zip(("train", "val", "test"), dm.splits())}, indent=2))

    trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)
    if not args.smoke and trainer.state.finished:
        trainer.test(model, datamodule=dm, ckpt_path="best")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
