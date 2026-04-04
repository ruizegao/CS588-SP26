#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lidar_det.config import AppConfig
from lidar_det.data.processed import ProcessedBEVDataset, ProcessedSplitStore, collate_processed_batch
from lidar_det.data.targets import decode_predictions
from lidar_det.eval.metrics import evaluate_map, plot_pr_curves
from lidar_det.models.simple_centerpoint import SimpleCenterPoint, compute_losses
from lidar_det.ops.nms import class_aware_nms
from lidar_det.train import resolve_device


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for processed-split evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on a processed split.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--split", default="val")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--score-thresh", type=float, default=None)
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--nms-iou", type=float, default=None)
    parser.add_argument("--iou-threshold", type=float, default=None)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--out-dir", default="outputs/processed_eval")
    return parser.parse_args()


def _load_checkpoint_model(checkpoint_path: str, device: torch.device) -> tuple[SimpleCenterPoint, AppConfig]:
    """Load a model checkpoint and reconstruct the model config."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = AppConfig.from_dict(ckpt["config"])
    model = SimpleCenterPoint(
        in_channels=len(cfg.bev.channels),
        num_classes=len(cfg.data.class_names),
        base_ch=cfg.model.base_channels,
        use_batchnorm=cfg.model.use_batchnorm,
        extra_res_blocks=cfg.model.extra_res_blocks,
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, cfg


def _run_eval(
    model: torch.nn.Module,
    loader: DataLoader,
    cfg: AppConfig,
    device: torch.device,
) -> Dict[str, object]:
    """Run split evaluation and collect losses plus mAP statistics."""
    loss_acc = {"total": 0.0, "heatmap": 0.0, "reg": 0.0, "height": 0.0, "dims": 0.0, "rot": 0.0}
    num_batches = 0
    pred_by_frame: Dict[int, Dict[str, np.ndarray]] = {}
    gt_by_frame: Dict[int, Dict[str, np.ndarray]] = {}

    with torch.no_grad():
        for batch in loader:
            bev = batch["bev"].to(device, non_blocking=True)
            targets = {k: v.to(device, non_blocking=True) for k, v in batch["targets"].items()}
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
                    rotated=cfg.infer.use_rotated_nms,
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

    eval_result = evaluate_map(
        predictions=pred_by_frame,
        ground_truth=gt_by_frame,
        class_names=cfg.data.class_names,
        iou_threshold=cfg.eval.iou_threshold,
        rotated_iou=cfg.infer.use_rotated_nms,
    )
    denom = max(num_batches, 1)
    return {
        "loss_total": loss_acc["total"] / denom,
        "loss_heatmap": loss_acc["heatmap"] / denom,
        "loss_reg": loss_acc["reg"] / denom,
        "loss_height": loss_acc["height"] / denom,
        "loss_dims": loss_acc["dims"] / denom,
        "loss_rot": loss_acc["rot"] / denom,
        "mAP": eval_result["mAP"],
        "eval_result": eval_result,
    }


def main() -> None:
    """Evaluate one processed split and write metrics plus PR curve to disk."""
    args = parse_args()
    device = resolve_device(args.device)
    model, cfg = _load_checkpoint_model(args.checkpoint, device)

    if args.score_thresh is not None:
        cfg.infer.score_threshold = args.score_thresh
    if args.topk is not None:
        cfg.infer.topk = args.topk
    if args.nms_iou is not None:
        cfg.infer.nms_iou_threshold = args.nms_iou
    if args.iou_threshold is not None:
        cfg.eval.iou_threshold = args.iou_threshold
    cfg.device = args.device

    cache_dir = args.cache_dir or str(Path(args.processed_dir) / "cache")
    store = ProcessedSplitStore(args.processed_dir, args.split, cache_dir=cache_dir)
    sample_indices = None if args.max_samples < 0 else list(range(min(args.max_samples, len(store))))
    dataset = ProcessedBEVDataset(store, sample_indices=sample_indices, include_targets=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_processed_batch,
        drop_last=False,
    )

    stats = _run_eval(model, loader, cfg, device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / f"{args.split}_metrics.txt"
    pr_path = out_dir / f"{args.split}_pr_curve.png"
    plot_pr_curves(stats["eval_result"], str(pr_path))

    lines = [
        f"split={args.split}",
        f"checkpoint={args.checkpoint}",
        f"score_thresh={cfg.infer.score_threshold}",
        f"topk={cfg.infer.topk}",
        f"nms_iou={cfg.infer.nms_iou_threshold}",
        f"iou_threshold={cfg.eval.iou_threshold}",
        f"loss_total={stats['loss_total']:.6f}",
        f"loss_heatmap={stats['loss_heatmap']:.6f}",
        f"loss_reg={stats['loss_reg']:.6f}",
        f"loss_height={stats['loss_height']:.6f}",
        f"loss_dims={stats['loss_dims']:.6f}",
        f"loss_rot={stats['loss_rot']:.6f}",
        f"mAP={stats['mAP']:.6f}",
        f"pr_curve={pr_path}",
    ]
    for class_name, result in stats["eval_result"]["per_class"].items():
        lines.append(
            f"class={class_name} AP={result['ap']:.6f} num_gt={result['num_gt']} num_det={result['num_det']}"
        )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
