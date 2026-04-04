#!/usr/bin/env python
from __future__ import annotations

import argparse
from datetime import datetime
import json
import time
from pathlib import Path
import sys
from typing import Dict, IO, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lidar_det.config import default_config
from lidar_det.data.processed import ProcessedBEVDataset, ProcessedSplitStore, collate_processed_batch
from lidar_det.data.targets import decode_predictions
from lidar_det.eval.metrics import evaluate_map, plot_pr_curves
from lidar_det.models.simple_centerpoint import SimpleCenterPoint, compute_losses, train_step
from lidar_det.ops.nms import class_aware_nms
from lidar_det.train import resolve_device, seed_everything


class _TimestampedTee:
    """Mirror writes to the terminal and a log file with wall-clock and elapsed-time prefixes."""

    def __init__(self, terminal: IO[str], log_file: IO[str], start_time: float) -> None:
        self._terminal = terminal
        self._log_file = log_file
        self._start_time = start_time
        self._pending = ""

    @property
    def encoding(self) -> str:
        """Expose the wrapped stream encoding."""
        return getattr(self._terminal, "encoding", "utf-8")

    def isatty(self) -> bool:
        """Preserve TTY behavior for downstream tools."""
        return bool(getattr(self._terminal, "isatty", lambda: False)())

    def fileno(self) -> int:
        """Forward file descriptor access when available."""
        return self._terminal.fileno()

    def write(self, text: str) -> int:
        """Write text, prefixing each completed line before teeing it."""
        if not text:
            return 0
        self._pending += text
        while "\n" in self._pending:
            line, self._pending = self._pending.split("\n", 1)
            self._emit(line)
        return len(text)

    def flush(self) -> None:
        """Flush the wrapped streams."""
        self._terminal.flush()
        self._log_file.flush()

    def close(self) -> None:
        """Flush any trailing partial line."""
        if self._pending:
            self._emit(self._pending)
            self._pending = ""
        self.flush()

    def _emit(self, line: str) -> None:
        """Emit one fully formatted line."""
        wall = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elapsed = time.perf_counter() - self._start_time
        prefix = f"[{wall} +{elapsed:8.1f}s] "
        payload = prefix + line + "\n"
        self._terminal.write(payload)
        self._log_file.write(payload)
        self._terminal.flush()
        self._log_file.flush()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for processed-split training."""
    parser = argparse.ArgumentParser(description="Train a CenterPoint-style detector on processed NPZ splits.")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--val-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--extra-res-blocks", type=int, default=0)
    parser.add_argument("--batchnorm", dest="use_batchnorm", action="store_true")
    parser.add_argument("--no-batchnorm", dest="use_batchnorm", action="store_false")
    parser.set_defaults(use_batchnorm=True)
    parser.add_argument("--optimizer", choices=["adam", "adamw"], default="adam")
    parser.add_argument("--lr-scheduler", choices=["fixed", "cosine"], default="fixed")

    parser.add_argument("--score-thresh", type=float, default=0.2)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--nms-iou", type=float, default=0.1)
    parser.add_argument("--eval-iou-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--resume", default="")
    parser.add_argument("--resume-optimizer", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=-1)
    parser.add_argument("--max-val-samples", type=int, default=-1)
    parser.add_argument("--max-test-samples", type=int, default=-1)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--tensorboard", dest="tensorboard", action="store_true")
    parser.add_argument("--no-tensorboard", dest="tensorboard", action="store_false")
    parser.set_defaults(tensorboard=True)
    parser.add_argument("--tensorboard-dir", default="outputs/tensorboard")
    parser.add_argument("--eval-test-every-epoch", action="store_true")
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--fast-eval-axis-aligned", action="store_true")
    parser.add_argument("--eval-log-every", type=int, default=0)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--out-dir", default="outputs/processed_train")
    return parser.parse_args()


def _subset_indices(total: int, limit: int) -> Optional[List[int]]:
    """Return the leading subset indices for a split when a limit is requested."""
    if limit is None or limit < 0 or limit >= total:
        return None
    return list(range(limit))


def _build_cfg(store: ProcessedSplitStore, args: argparse.Namespace):
    """Create an application config aligned with the processed dataset."""
    cfg = default_config()
    for key, value in store.metadata.get("bev_config", {}).items():
        if hasattr(cfg.bev, key):
            setattr(cfg.bev, key, value)
    for key, value in store.metadata.get("target_config", {}).items():
        if hasattr(cfg.target, key):
            setattr(cfg.target, key, value)
    cfg.data.class_names = list(store.metadata["class_names"])
    cfg.model.base_channels = args.base_channels
    cfg.model.use_batchnorm = args.use_batchnorm

    cfg.model.extra_res_blocks = args.extra_res_blocks
    cfg.train.batch_size = args.batch_size
    cfg.train.val_batch_size = args.val_batch_size
    cfg.train.num_workers = args.num_workers
    cfg.train.learning_rate = args.lr
    cfg.train.weight_decay = args.weight_decay
    cfg.train.log_every = args.log_every
    cfg.train.seed = args.seed
    cfg.infer.score_threshold = args.score_thresh
    cfg.infer.topk = args.topk
    cfg.infer.nms_iou_threshold = args.nms_iou
    cfg.eval.iou_threshold = args.eval_iou_threshold
    cfg.device = args.device
    return cfg


def _build_loader(
    dataset: ProcessedBEVDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    """Build a dataloader for processed samples."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_processed_batch,
        drop_last=False,
    )


def _save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, cfg) -> None:
    """Save a training checkpoint with config and optimizer state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "config": cfg.to_dict(),
        },
        path,
    )


def _load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, object]:
    """Load a checkpoint into the model and optionally the optimizer."""
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt


def _run_eval_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    cfg,
    device: torch.device,
    *,
    rotated_nms: Optional[bool] = None,
    rotated_iou: Optional[bool] = None,
    log_prefix: str = "",
    log_every: int = 0,
) -> Dict[str, object]:
    """Evaluate one split and return losses, predictions, and metrics."""
    model.eval()
    loss_acc = {"total": 0.0, "heatmap": 0.0, "reg": 0.0, "height": 0.0, "dims": 0.0, "rot": 0.0}
    num_batches = 0
    pred_by_frame: Dict[int, Dict[str, np.ndarray]] = {}
    gt_by_frame: Dict[int, Dict[str, np.ndarray]] = {}
    rotated_nms = cfg.infer.use_rotated_nms if rotated_nms is None else rotated_nms
    rotated_iou = cfg.infer.use_rotated_nms if rotated_iou is None else rotated_iou
    eval_start = time.perf_counter()

    with torch.no_grad():
        for batch in loader:
            bev = batch["bev"].to(device)
            targets = {k: v.to(device) for k, v in batch["targets"].items()}
            outputs = model(bev)
            loss_dict = compute_losses(outputs, targets, cfg.train)
            for key in loss_acc:
                loss_acc[key] += float(loss_dict[key].item())
            num_batches += 1

            decoded = decode_predictions(
                outputs,
                bev_cfg=cfg.bev,
                target_cfg=cfg.target,
                score_threshold=cfg.infer.score_threshold,
                topk=cfg.infer.topk,
            )

            sample_ids = batch["frame_idx"].cpu().numpy().tolist()
            for b_idx, sample_id in enumerate(sample_ids):
                pred_boxes, pred_scores, pred_classes = class_aware_nms(
                    decoded[b_idx]["boxes"],
                    decoded[b_idx]["scores"],
                    decoded[b_idx]["classes"],
                    iou_threshold=cfg.infer.nms_iou_threshold,
                    max_detections=cfg.infer.max_detections,
                    rotated=rotated_nms,
                )
                pred_by_frame[int(sample_id)] = {
                    "boxes": pred_boxes,
                    "scores": pred_scores,
                    "classes": pred_classes,
                }
                gt_boxes = batch["gt_boxes"][b_idx].cpu().numpy().astype(np.float32)
                gt_classes = batch["gt_classes"][b_idx].cpu().numpy().astype(np.int64)
                gt_by_frame[int(sample_id)] = {
                    "boxes": gt_boxes,
                    "scores": np.ones((gt_boxes.shape[0],), dtype=np.float32),
                    "classes": gt_classes,
                }
            if log_every > 0 and (num_batches % log_every == 0 or num_batches == 1):
                print(
                    f"{log_prefix}eval_batch={num_batches:04d}/{len(loader):04d} "
                    f"elapsed_s={time.perf_counter() - eval_start:.2f}"
                )

    denom = max(num_batches, 1)
    avg_losses = {k: v / denom for k, v in loss_acc.items()}
    if log_prefix:
        print(
            f"{log_prefix}metric_stage_start frames={len(pred_by_frame)} "
            f"elapsed_s={time.perf_counter() - eval_start:.2f} rotated_iou={rotated_iou}"
        )
    eval_result = evaluate_map(
        predictions=pred_by_frame,
        ground_truth=gt_by_frame,
        class_names=cfg.data.class_names,
        iou_threshold=cfg.eval.iou_threshold,
        rotated_iou=rotated_iou,
    )
    model.train()
    return {
        "loss_total": float(avg_losses["total"]),
        "loss_heatmap": float(avg_losses["heatmap"]),
        "loss_reg": float(avg_losses["reg"]),
        "loss_height": float(avg_losses["height"]),
        "loss_dims": float(avg_losses["dims"]),
        "loss_rot": float(avg_losses["rot"]),
        "eval_result": eval_result,
        "mAP": float(eval_result["mAP"]),
        "eval_time_s": float(time.perf_counter() - eval_start),
        "predictions": pred_by_frame,
        "ground_truth": gt_by_frame,
    }


def _plot_history(history: List[Dict[str, float]], save_path: Path) -> None:
    """Plot train and val curves to a PNG file."""
    epochs = [row["epoch"] for row in history]
    def _series(key: str) -> List[float]:
        values: List[float] = []
        for row in history:
            value = row.get(key, None)
            values.append(np.nan if value is None else float(value))
        return values

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(epochs, _series("train_total"), label="train_total", linewidth=2)
    axes[0].plot(epochs, _series("val_total"), label="val_total", linewidth=2)
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Total Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    for key in ["heatmap", "reg", "height", "dims", "rot"]:
        axes[1].plot(epochs, _series(f"train_{key}"), label=f"train_{key}")
        axes[1].plot(epochs, _series(f"val_{key}"), linestyle="--", label=f"val_{key}")
    axes[1].plot(epochs, _series("val_mAP"), label="val_mAP", linewidth=2, color="black")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Metric")
    axes[1].set_title("Per-Term Losses and Val mAP")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(ncol=3, fontsize=8)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def _write_eval_summary(path: Path, split_name: str, eval_stats: Dict[str, object], epoch_time_s: Optional[float] = None) -> None:
    """Write a text summary of evaluation metrics and losses."""
    lines = [
        f"split={split_name}",
        f"loss_total={eval_stats['loss_total']:.6f}",
        f"loss_heatmap={eval_stats['loss_heatmap']:.6f}",
        f"loss_reg={eval_stats['loss_reg']:.6f}",
        f"loss_height={eval_stats['loss_height']:.6f}",
        f"loss_dims={eval_stats['loss_dims']:.6f}",
        f"loss_rot={eval_stats['loss_rot']:.6f}",
        f"mAP={eval_stats['mAP']:.6f}",
    ]
    if epoch_time_s is not None:
        lines.append(f"epoch_time_s={epoch_time_s:.3f}")
    for class_name, result in eval_stats["eval_result"]["per_class"].items():
        lines.append(
            f"class={class_name} AP={result['ap']:.6f} num_gt={result['num_gt']} num_det={result['num_det']}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Train on processed splits, save checkpoints, curves, and eval summaries."""
    args = parse_args()
    run_name = args.run_name or f"proc_e{args.epochs}_bs{args.batch_size}_lr{args.lr:.0e}_{int(time.time())}"
    run_dir = Path(args.out_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    script_start = time.perf_counter()
    log_path = run_dir / "stdout.log.txt"
    log_file = log_path.open("a", encoding="utf-8", buffering=1)
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    stdout_tee = _TimestampedTee(orig_stdout, log_file, script_start)
    stderr_tee = _TimestampedTee(orig_stderr, log_file, script_start)
    sys.stdout = stdout_tee
    sys.stderr = stderr_tee

    try:
        seed_everything(args.seed)
        torch.set_float32_matmul_precision("high")

        cache_dir = args.cache_dir or str(Path(args.processed_dir) / "cache")
        train_store = ProcessedSplitStore(args.processed_dir, args.train_split, cache_dir=cache_dir)
        val_store = ProcessedSplitStore(args.processed_dir, args.val_split, cache_dir=cache_dir)
        test_store = ProcessedSplitStore(args.processed_dir, args.test_split, cache_dir=cache_dir)

        cfg = _build_cfg(train_store, args)
        cfg_dump_path = run_dir / "config.json"
        cfg_dump_path.write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
        device = resolve_device(args.device)

        train_dataset = ProcessedBEVDataset(
            train_store,
            sample_indices=_subset_indices(len(train_store), args.max_train_samples),
            include_targets=True,
        )
        val_dataset = ProcessedBEVDataset(
            val_store,
            sample_indices=_subset_indices(len(val_store), args.max_val_samples),
            include_targets=True,
        )
        test_dataset = ProcessedBEVDataset(
            test_store,
            sample_indices=_subset_indices(len(test_store), args.max_test_samples),
            include_targets=True,
        )

        train_loader = _build_loader(train_dataset, args.batch_size, True, args.num_workers, device)
        val_loader = _build_loader(val_dataset, args.val_batch_size, False, args.num_workers, device)
        test_loader = _build_loader(test_dataset, args.val_batch_size, False, args.num_workers, device)

        print(
            f"run_name={run_name} device={device} optimizer={args.optimizer} scheduler={args.lr_scheduler} "
            f"batchnorm={args.use_batchnorm} eval_every={args.eval_every}"
        )
        print(
            f"train_split={args.train_split} samples={len(train_dataset)} val_split={args.val_split} samples={len(val_dataset)} "
            f"test_split={args.test_split} samples={len(test_dataset)} batch_size={args.batch_size} val_batch_size={args.val_batch_size}"
        )
        print(f"stdout_log={log_path}")

        model = SimpleCenterPoint(
            in_channels=len(train_store.input_channels),
            num_classes=train_store.num_classes,
            base_ch=cfg.model.base_channels,
            use_batchnorm=cfg.model.use_batchnorm,
            extra_res_blocks=cfg.model.extra_res_blocks,
        ).to(device)
        if args.optimizer == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = None
        if args.lr_scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
        resumed_epoch = 0
        if args.resume:
            ckpt = _load_checkpoint(args.resume, model, optimizer if args.resume_optimizer else None)
            resumed_epoch = int(ckpt.get("epoch", 0))
            print(
                f"resumed_from={args.resume} resumed_epoch={resumed_epoch} "
                f"resume_optimizer={bool(args.resume_optimizer)}"
            )
        writer = None
        tensorboard_log_dir = ""
        if args.tensorboard and SummaryWriter is not None:
            tensorboard_log_dir = str(Path(args.tensorboard_dir) / run_name)
            writer = SummaryWriter(log_dir=tensorboard_log_dir)
            print(f"tensorboard_log_dir={tensorboard_log_dir}")
        elif args.tensorboard and SummaryWriter is None:
            print("TensorBoard disabled: torch.utils.tensorboard.SummaryWriter is unavailable.")

        history: List[Dict[str, float]] = []
        best_val_map = -1.0
        best_ckpt_path = run_dir / "best_val.pt"

        total_train_start = time.perf_counter()
        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_start = time.perf_counter()
            train_loss_acc = {"total": 0.0, "heatmap": 0.0, "reg": 0.0, "height": 0.0, "dims": 0.0, "rot": 0.0}
            num_batches = 0

            for step_idx, batch in enumerate(train_loader, start=1):
                global_step = (epoch - 1) * max(len(train_loader), 1) + step_idx
                bev = batch["bev"].to(device, non_blocking=True)
                targets = {k: v.to(device, non_blocking=True) for k, v in batch["targets"].items()}

                loss_dict = train_step(
                    model=model,
                    bev=bev,
                    targets=targets,
                    optimizer=optimizer,
                    train_cfg=cfg.train,
                )

                for key in train_loss_acc:
                    train_loss_acc[key] += float(loss_dict[key].item())
                num_batches += 1

                if writer is not None:
                    writer.add_scalar("train_step/total", float(loss_dict["total"].item()), global_step)
                    writer.add_scalar("train_step/heatmap", float(loss_dict["heatmap"].item()), global_step)
                    writer.add_scalar("train_step/reg", float(loss_dict["reg"].item()), global_step)
                    writer.add_scalar("train_step/height", float(loss_dict["height"].item()), global_step)
                    writer.add_scalar("train_step/dims", float(loss_dict["dims"].item()), global_step)
                    writer.add_scalar("train_step/rot", float(loss_dict["rot"].item()), global_step)
                    writer.add_scalar("train_step/lr", float(optimizer.param_groups[0]["lr"]), global_step)

                if step_idx % max(1, args.log_every) == 0 or step_idx == 1:
                    print(
                        f"epoch={epoch:02d} step={step_idx:04d}/{len(train_loader):04d} "
                        f"total={loss_dict['total'].item():.4f} hm={loss_dict['heatmap'].item():.4f} "
                        f"reg={loss_dict['reg'].item():.4f} z={loss_dict['height'].item():.4f} "
                        f"dims={loss_dict['dims'].item():.4f} yaw={loss_dict['rot'].item():.4f}"
                    )

            if scheduler is not None:
                scheduler.step()
            epoch_time_s = time.perf_counter() - epoch_start
            train_avg = {key: value / max(num_batches, 1) for key, value in train_loss_acc.items()}
            eval_every = max(1, int(args.eval_every))
            should_eval = (epoch % eval_every == 0) or (epoch == args.epochs)

            epoch_row = {
                "epoch": epoch,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "epoch_time_s": epoch_time_s,
                "train_total": train_avg["total"],
                "train_heatmap": train_avg["heatmap"],
                "train_reg": train_avg["reg"],
                "train_height": train_avg["height"],
                "train_dims": train_avg["dims"],
                "train_rot": train_avg["rot"],
            }
            if should_eval:
                val_stats = _run_eval_epoch(
                    model,
                    val_loader,
                    cfg,
                    device,
                    rotated_nms=False if args.fast_eval_axis_aligned else None,
                    rotated_iou=False if args.fast_eval_axis_aligned else None,
                    log_prefix="val_",
                    log_every=args.eval_log_every,
                )
                print(
                    f"val epoch={epoch:02d} total={val_stats['loss_total']:.4f} "
                    f"hm={val_stats['loss_heatmap']:.4f} reg={val_stats['loss_reg']:.4f} "
                    f"z={val_stats['loss_height']:.4f} dims={val_stats['loss_dims']:.4f} "
                    f"yaw={val_stats['loss_rot']:.4f} mAP={val_stats['mAP']:.4f} "
                    f"train_time_s={epoch_time_s:.2f} val_time_s={val_stats['eval_time_s']:.2f}"
                )
                epoch_row["val_time_s"] = val_stats["eval_time_s"]
                epoch_row["val_total"] = val_stats["loss_total"]
                epoch_row["val_heatmap"] = val_stats["loss_heatmap"]
                epoch_row["val_reg"] = val_stats["loss_reg"]
                epoch_row["val_height"] = val_stats["loss_height"]
                epoch_row["val_dims"] = val_stats["loss_dims"]
                epoch_row["val_rot"] = val_stats["loss_rot"]
                epoch_row["val_mAP"] = val_stats["mAP"]

                if args.eval_test_every_epoch:
                    test_stats = _run_eval_epoch(
                        model,
                        test_loader,
                        cfg,
                        device,
                        rotated_nms=False if args.fast_eval_axis_aligned else None,
                        rotated_iou=False if args.fast_eval_axis_aligned else None,
                        log_prefix="test_",
                        log_every=args.eval_log_every,
                    )
                    print(
                        f"test epoch={epoch:02d} total={test_stats['loss_total']:.4f} "
                        f"hm={test_stats['loss_heatmap']:.4f} reg={test_stats['loss_reg']:.4f} "
                        f"z={test_stats['loss_height']:.4f} dims={test_stats['loss_dims']:.4f} "
                        f"yaw={test_stats['loss_rot']:.4f} mAP={test_stats['mAP']:.4f} "
                        f"test_time_s={test_stats['eval_time_s']:.2f}"
                    )
                    epoch_row["test_mAP"] = test_stats["mAP"]
                    epoch_row["test_time_s"] = test_stats["eval_time_s"]
                    _write_eval_summary(
                        run_dir / f"test_metrics_epoch_{epoch:02d}.txt",
                        args.test_split,
                        test_stats,
                        epoch_time_s=epoch_time_s,
                    )
                    _write_eval_summary(run_dir / "test_metrics_latest.txt", args.test_split, test_stats, epoch_time_s=epoch_time_s)
            else:
                print(f"val epoch={epoch:02d} skipped exact eval (eval_every={eval_every})")

            history.append(epoch_row)

            if writer is not None:
                writer.add_scalar("train_epoch/total", train_avg["total"], epoch)
                writer.add_scalar("train_epoch/heatmap", train_avg["heatmap"], epoch)
                writer.add_scalar("train_epoch/reg", train_avg["reg"], epoch)
                writer.add_scalar("train_epoch/height", train_avg["height"], epoch)
                writer.add_scalar("train_epoch/dims", train_avg["dims"], epoch)
                writer.add_scalar("train_epoch/rot", train_avg["rot"], epoch)
                writer.add_scalar("train_epoch/lr", float(optimizer.param_groups[0]["lr"]), epoch)
                if should_eval:
                    writer.add_scalar("val_epoch/total", val_stats["loss_total"], epoch)
                    writer.add_scalar("val_epoch/heatmap", val_stats["loss_heatmap"], epoch)
                    writer.add_scalar("val_epoch/reg", val_stats["loss_reg"], epoch)
                    writer.add_scalar("val_epoch/height", val_stats["loss_height"], epoch)
                    writer.add_scalar("val_epoch/dims", val_stats["loss_dims"], epoch)
                    writer.add_scalar("val_epoch/rot", val_stats["loss_rot"], epoch)
                    writer.add_scalar("val_epoch/mAP", val_stats["mAP"], epoch)
                if "test_mAP" in epoch_row:
                    writer.add_scalar("test_epoch/mAP", epoch_row["test_mAP"], epoch)

            _save_checkpoint(run_dir / "last.pt", model, optimizer, epoch, cfg)
            if should_eval and val_stats["mAP"] > best_val_map:
                best_val_map = val_stats["mAP"]
                _save_checkpoint(best_ckpt_path, model, optimizer, epoch, cfg)
                print(f"new_best_val_map={best_val_map:.4f} checkpoint={best_ckpt_path}")

            (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            _plot_history(history, run_dir / "loss_curves.png")
            if should_eval:
                _write_eval_summary(run_dir / f"val_metrics_epoch_{epoch:02d}.txt", args.val_split, val_stats, epoch_time_s=epoch_time_s)
                _write_eval_summary(run_dir / "val_metrics.txt", args.val_split, val_stats, epoch_time_s=epoch_time_s)

        train_wall_time_s = time.perf_counter() - total_train_start

        best_ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(best_ckpt["model"])
        final_val_stats = _run_eval_epoch(model, val_loader, cfg, device)
        final_test_stats = _run_eval_epoch(model, test_loader, cfg, device)

        plot_pr_curves(final_val_stats["eval_result"], str(run_dir / "val_pr_curve.png"))
        plot_pr_curves(final_test_stats["eval_result"], str(run_dir / "test_pr_curve.png"))
        _write_eval_summary(run_dir / "val_metrics_best.txt", args.val_split, final_val_stats)
        _write_eval_summary(run_dir / "test_metrics.txt", args.test_split, final_test_stats)

        summary = {
            "run_name": run_name,
            "train_split": args.train_split,
            "val_split": args.val_split,
            "test_split": args.test_split,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "optimizer": args.optimizer,
            "lr_scheduler": args.lr_scheduler,
            "resume_checkpoint": args.resume,
            "resume_optimizer": bool(args.resume_optimizer),
            "eval_every": int(args.eval_every),
            "fast_eval_axis_aligned": bool(args.fast_eval_axis_aligned),
            "device": str(device),
            "tensorboard_log_dir": tensorboard_log_dir,
            "stdout_log": str(log_path),
            "train_wall_time_s": train_wall_time_s,
            "best_val_mAP": final_val_stats["mAP"],
            "test_mAP": final_test_stats["mAP"],
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        if writer is not None:
            writer.flush()
            writer.close()

        print(f"run_dir={run_dir}")
        print(f"best_val_mAP={final_val_stats['mAP']:.6f}")
        print(f"test_mAP={final_test_stats['mAP']:.6f}")
        print(f"train_wall_time_s={train_wall_time_s:.3f}")
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        stdout_tee.close()
        stderr_tee.close()
        log_file.close()


if __name__ == "__main__":
    main()
