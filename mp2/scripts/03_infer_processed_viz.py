#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lidar_det.config import AppConfig
from lidar_det.data.processed import ProcessedBEVDataset, ProcessedSplitStore
from lidar_det.data.targets import decode_predictions
from lidar_det.eval.iou import iou_bev
from lidar_det.models.simple_centerpoint import SimpleCenterPoint
from lidar_det.ops.nms import class_aware_nms
from lidar_det.train import resolve_device
from lidar_det.viz.bev_plot import render_bev_with_boxes


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for processed-split visualization."""
    parser = argparse.ArgumentParser(description="Visualize predictions on processed split debug frames.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--raw-root", default="data")
    parser.add_argument("--seq-date", default="2011_09_26")
    parser.add_argument("--split", default="minival")
    parser.add_argument("--score-thresh", type=float, default=None)
    parser.add_argument("--no-auto-score-thresh-f1", action="store_true")
    parser.add_argument("--score-thresh-min", type=float, default=0.0)
    parser.add_argument("--score-thresh-max", type=float, default=0.5)
    parser.add_argument("--score-thresh-step", type=float, default=0.01)
    parser.add_argument("--f1-iou-thresh", type=float, default=0.5)
    parser.add_argument("--f1-axis-aligned", action="store_true")
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--nms-iou", type=float, default=None)
    parser.add_argument("--max-debug-frames", type=int, default=10)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--out-dir", default="outputs/processed_viz")
    return parser.parse_args()


def _parse_calib_file(path: Path) -> Dict[str, np.ndarray]:
    """Parse a KITTI calibration text file into a numeric dictionary."""
    data: Dict[str, np.ndarray] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            vals = value.strip().split()
            if not vals:
                continue
            try:
                nums = np.array([float(v) for v in vals], dtype=np.float64)
            except ValueError:
                continue
            data[key] = nums
    return data


def _load_cam2_projection(raw_root: Path, seq_date: str) -> np.ndarray:
    """Load the LiDAR-to-camera-2 projection matrix for one KITTI date."""
    date_dir = raw_root / seq_date
    cam_calib = _parse_calib_file(date_dir / "calib_cam_to_cam.txt")
    velo_calib = _parse_calib_file(date_dir / "calib_velo_to_cam.txt")

    tr = np.eye(4, dtype=np.float64)
    tr[:3, :3] = velo_calib["R"].reshape(3, 3)
    tr[:3, 3] = velo_calib["T"]

    r_rect = np.eye(4, dtype=np.float64)
    r_rect[:3, :3] = cam_calib["R_rect_00"].reshape(3, 3)

    p_rect_02 = cam_calib["P_rect_02"].reshape(3, 4)
    return p_rect_02 @ r_rect @ tr


def _load_raw_frame_from_disk(
    raw_root: Path,
    seq_date: str,
    drive_id: int,
    frame_idx: int,
) -> Dict[str, object]:
    """Load one raw KITTI frame directly from disk using drive and frame identifiers."""
    drive_dir = raw_root / seq_date / f"{seq_date}_drive_{int(drive_id):04d}_sync"
    image_path = drive_dir / "image_02" / "data" / f"{int(frame_idx):010d}.png"
    points_path = drive_dir / "velodyne_points" / "data" / f"{int(frame_idx):010d}.bin"

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Missing image_02 frame: {image_path}")
    points = np.fromfile(points_path, dtype=np.float32).reshape(-1, 4)
    return {
        "seq_date": seq_date,
        "drive_id": int(drive_id),
        "frame_idx": int(frame_idx),
        "image_02": cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
        "points": points,
    }


def _box_to_corners_lidar(box: np.ndarray) -> np.ndarray:
    """Convert one [x, y, z, l, w, h, yaw] box to its 8 LiDAR-frame corners."""
    x, y, z, l, w, h, yaw = [float(v) for v in box]
    corners = np.array(
        [
            [l / 2, w / 2, h / 2],
            [l / 2, -w / 2, h / 2],
            [-l / 2, -w / 2, h / 2],
            [-l / 2, w / 2, h / 2],
            [l / 2, w / 2, -h / 2],
            [l / 2, -w / 2, -h / 2],
            [-l / 2, -w / 2, -h / 2],
            [-l / 2, w / 2, -h / 2],
        ],
        dtype=np.float32,
    )
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    rot = np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return (rot @ corners.T).T + np.array([x, y, z], dtype=np.float32)


def _project_points(points_xyz: np.ndarray, proj: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Project LiDAR-frame 3D points to image coordinates and depth."""
    pts_h = np.concatenate([points_xyz.astype(np.float64), np.ones((points_xyz.shape[0], 1), dtype=np.float64)], axis=1)
    uvw = (proj @ pts_h.T).T
    depth = uvw[:, 2]
    valid = depth > 0.1
    proj_xy = np.zeros((points_xyz.shape[0], 2), dtype=np.float32)
    proj_xy[valid, 0] = (uvw[valid, 0] / depth[valid]).astype(np.float32)
    proj_xy[valid, 1] = (uvw[valid, 1] / depth[valid]).astype(np.float32)
    return proj_xy, valid


def _draw_projected_boxes(image: np.ndarray, boxes: np.ndarray, proj: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
    """Draw projected 3D box wireframes on an image."""
    out = image.copy()
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for box in boxes:
        corners = _box_to_corners_lidar(box)
        proj_xy, valid = _project_points(corners, proj)
        if not np.all(valid):
            continue
        pts = proj_xy.astype(np.int32)
        for start, end in edges:
            cv2.line(out, tuple(pts[start]), tuple(pts[end]), color, 2, lineType=cv2.LINE_AA)
    return out


def _overlay_points(image: np.ndarray, points: np.ndarray, proj: np.ndarray) -> np.ndarray:
    """Overlay projected LiDAR points on an RGB image."""
    out = image.copy()
    proj_xy, valid = _project_points(points[:, :3], proj)
    depth = points[:, 0]
    proj_xy = proj_xy[valid]
    depth = depth[valid]
    if proj_xy.shape[0] == 0:
        return out

    h, w = out.shape[:2]
    depth_lo = float(np.percentile(depth, 5))
    depth_hi = float(np.percentile(depth, 95))
    denom = max(depth_hi - depth_lo, 1e-6)
    colors = np.clip((depth - depth_lo) / denom, 0.0, 1.0)
    colors = (plt_cm_inferno(colors)[:, :3] * 255.0).astype(np.uint8)
    pts = proj_xy.astype(np.int32)
    mask = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
    for (u, v), color in zip(pts[mask][::2], colors[mask][::2]):
        cv2.circle(out, (int(u), int(v)), 2, color.tolist(), -1, lineType=cv2.LINE_AA)
    return out


def plt_cm_inferno(values: np.ndarray) -> np.ndarray:
    """Map scalar values in [0, 1] to an inferno colormap without importing pyplot globally."""
    import matplotlib.cm as mpl_cm

    return mpl_cm.get_cmap("inferno")(values)


def _load_model(checkpoint_path: str, device: torch.device) -> Tuple[SimpleCenterPoint, AppConfig]:
    """Load the trained model and stored config."""
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


def _empty_pred() -> Dict[str, np.ndarray]:
    """Return an empty prediction record."""
    return {
        "boxes": np.zeros((0, 7), dtype=np.float32),
        "scores": np.zeros((0,), dtype=np.float32),
        "classes": np.zeros((0,), dtype=np.int64),
    }


def _filter_predictions_by_score(
    pred_by_frame: Dict[int, Dict[str, np.ndarray]],
    score_thresh: float,
) -> Dict[int, Dict[str, np.ndarray]]:
    """Filter predictions by a score threshold without changing frame ids."""
    out: Dict[int, Dict[str, np.ndarray]] = {}
    for frame_idx, pred in pred_by_frame.items():
        keep = pred["scores"] >= float(score_thresh)
        out[frame_idx] = {
            "boxes": pred["boxes"][keep],
            "scores": pred["scores"][keep],
            "classes": pred["classes"][keep],
        }
    return out


def _compute_detection_f1(
    pred_by_frame: Dict[int, Dict[str, np.ndarray]],
    gt_by_frame: Dict[int, Dict[str, np.ndarray]],
    iou_thresh: float,
    rotated_iou: bool,
) -> Dict[str, float]:
    """Compute precision, recall, and F1 across a frame-indexed prediction set."""
    tp = 0
    fp = 0
    fn = 0

    frame_ids = sorted(set(pred_by_frame.keys()) | set(gt_by_frame.keys()))
    for frame_idx in frame_ids:
        pred = pred_by_frame.get(frame_idx, _empty_pred())
        gt = gt_by_frame.get(frame_idx, _empty_pred())
        pred_boxes = pred["boxes"]
        pred_scores = pred["scores"]
        pred_classes = pred["classes"]
        gt_boxes = gt["boxes"]
        gt_classes = gt["classes"]

        all_classes = np.union1d(pred_classes, gt_classes)
        for cls_id in all_classes:
            cls_pred_mask = pred_classes == cls_id
            cls_gt_mask = gt_classes == cls_id
            cls_pred_boxes = pred_boxes[cls_pred_mask]
            cls_pred_scores = pred_scores[cls_pred_mask]
            cls_gt_boxes = gt_boxes[cls_gt_mask]

            if cls_pred_boxes.shape[0] == 0:
                fn += int(cls_gt_boxes.shape[0])
                continue
            if cls_gt_boxes.shape[0] == 0:
                fp += int(cls_pred_boxes.shape[0])
                continue

            order = np.argsort(-cls_pred_scores)
            cls_pred_boxes = cls_pred_boxes[order]
            matched_gt = np.zeros((cls_gt_boxes.shape[0],), dtype=bool)

            for pbox in cls_pred_boxes:
                best_iou = -1.0
                best_gt_idx = -1
                for gt_idx in range(cls_gt_boxes.shape[0]):
                    if matched_gt[gt_idx]:
                        continue
                    iou = iou_bev(pbox, cls_gt_boxes[gt_idx], rotated=rotated_iou)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                if best_iou >= iou_thresh and best_gt_idx >= 0:
                    tp += 1
                    matched_gt[best_gt_idx] = True
                else:
                    fp += 1
            fn += int((~matched_gt).sum())

    precision = tp / max(tp + fp, 1e-6)
    recall = tp / max(tp + fn, 1e-6)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-6)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }


def _select_best_f1_threshold(
    pred_by_frame: Dict[int, Dict[str, np.ndarray]],
    gt_by_frame: Dict[int, Dict[str, np.ndarray]],
    thresh_min: float,
    thresh_max: float,
    thresh_step: float,
    iou_thresh: float,
    rotated_iou: bool,
) -> Tuple[float, Dict[str, float]]:
    """Select the score threshold that maximizes F1 over the chosen frames."""
    step = max(float(thresh_step), 1e-6)
    thresholds = np.arange(float(thresh_min), float(thresh_max) + 0.5 * step, step)
    if thresholds.size == 0:
        thresholds = np.array([0.0], dtype=np.float32)

    best_thresh = float(thresholds[0])
    best_stats = {"f1": -1.0, "precision": -1.0, "recall": -1.0}
    for thresh in thresholds:
        filtered = _filter_predictions_by_score(pred_by_frame, float(thresh))
        stats = _compute_detection_f1(
            pred_by_frame=filtered,
            gt_by_frame=gt_by_frame,
            iou_thresh=iou_thresh,
            rotated_iou=rotated_iou,
        )
        is_better = (stats["f1"] > best_stats["f1"] + 1e-9) or (
            abs(stats["f1"] - best_stats["f1"]) <= 1e-9 and stats["precision"] > best_stats["precision"] + 1e-9
        )
        if is_better:
            best_thresh = float(thresh)
            best_stats = stats
    return best_thresh, best_stats


def main() -> None:
    """Render BEV and camera overlays for processed split debug frames."""
    args = parse_args()
    device = resolve_device(args.device)
    model, cfg = _load_model(args.checkpoint, device)
    use_auto_score_thresh = args.score_thresh is None and not args.no_auto_score_thresh_f1

    if use_auto_score_thresh:
        cfg.infer.score_threshold = 0.0
    elif args.score_thresh is not None:
        cfg.infer.score_threshold = args.score_thresh
    if args.topk is not None:
        cfg.infer.topk = args.topk
    if args.nms_iou is not None:
        cfg.infer.nms_iou_threshold = args.nms_iou

    cache_dir = args.cache_dir or str(Path(args.processed_dir) / "cache")
    store = ProcessedSplitStore(args.processed_dir, args.split, cache_dir=cache_dir)
    debug_indices = list(range(len(store)))
    if args.max_debug_frames > 0:
        debug_indices = debug_indices[: args.max_debug_frames]
    dataset = ProcessedBEVDataset(store, sample_indices=debug_indices, include_targets=False)

    proj_cam2 = _load_cam2_projection(Path(args.raw_root), args.seq_date)
    out_dir = Path(args.out_dir) / Path(args.checkpoint).stem / args.split
    bev_dir = out_dir / "bev"
    cam_dir = out_dir / "cam2"
    bev_dir.mkdir(parents=True, exist_ok=True)
    cam_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, object]] = []
    pred_by_frame: Dict[int, Dict[str, np.ndarray]] = {}
    gt_by_frame: Dict[int, Dict[str, np.ndarray]] = {}

    for local_idx, sample in enumerate(dataset):
        bev = sample["bev"].unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(bev)
            decoded = decode_predictions(
                outputs,
                bev_cfg=cfg.bev,
                target_cfg=cfg.target,
                score_threshold=cfg.infer.score_threshold,
                topk=cfg.infer.topk,
            )[0]
        pred_boxes, pred_scores, pred_classes = class_aware_nms(
            decoded["boxes"],
            decoded["scores"],
            decoded["classes"],
            iou_threshold=cfg.infer.nms_iou_threshold,
            max_detections=cfg.infer.max_detections,
            rotated=cfg.infer.use_rotated_nms,
        )

        sample_index = debug_indices[local_idx]
        drive_id = int(sample["drive_id"])
        frame_idx = int(sample["raw_frame_idx"])
        box_record = store.get_box_record(sample_index)
        raw_record = store.get_raw_record(sample_index)
        gt_boxes = np.asarray(box_record["boxes"], dtype=np.float32)
        gt_classes = np.asarray(box_record["class_ids"], dtype=np.int64)
        pred_by_frame[int(sample_index)] = {
            "boxes": pred_boxes,
            "scores": pred_scores,
            "classes": pred_classes,
        }
        gt_by_frame[int(sample_index)] = {
            "boxes": gt_boxes,
            "scores": np.ones((gt_boxes.shape[0],), dtype=np.float32),
            "classes": gt_classes,
        }
        records.append(
            {
                "sample_index": int(sample_index),
                "bev": sample["bev"].cpu().numpy(),
                "box_record": box_record,
                "raw_record": raw_record,
                "pred_boxes": pred_boxes,
                "pred_scores": pred_scores,
                "pred_classes": pred_classes,
            }
        )

    if use_auto_score_thresh:
        best_thresh, best_stats = _select_best_f1_threshold(
            pred_by_frame=pred_by_frame,
            gt_by_frame=gt_by_frame,
            thresh_min=args.score_thresh_min,
            thresh_max=args.score_thresh_max,
            thresh_step=args.score_thresh_step,
            iou_thresh=args.f1_iou_thresh,
            rotated_iou=not args.f1_axis_aligned,
        )
        cfg.infer.score_threshold = best_thresh
        stats_path = out_dir / "selected_threshold.txt"
        stats_path.write_text(
            (
                f"score_thresh={best_thresh:.6f}\n"
                f"f1={best_stats['f1']:.6f}\n"
                f"precision={best_stats['precision']:.6f}\n"
                f"recall={best_stats['recall']:.6f}\n"
                f"tp={int(best_stats['tp'])}\n"
                f"fp={int(best_stats['fp'])}\n"
                f"fn={int(best_stats['fn'])}\n"
            ),
            encoding="utf-8",
        )
        print(
            "auto_score_thresh_f1 selected threshold={:.4f} f1={:.4f} precision={:.4f} recall={:.4f} "
            "tp={} fp={} fn={}".format(
                best_thresh,
                best_stats["f1"],
                best_stats["precision"],
                best_stats["recall"],
                int(best_stats["tp"]),
                int(best_stats["fp"]),
                int(best_stats["fn"]),
            )
        )

    for record in records:
        keep = record["pred_scores"] >= float(cfg.infer.score_threshold)
        pred_boxes = record["pred_boxes"][keep]
        pred_scores = record["pred_scores"][keep]
        box_record = record["box_record"]
        raw_record = record["raw_record"]
        gt_boxes = np.asarray(box_record["boxes"], dtype=np.float32)
        bev_img = render_bev_with_boxes(
            bev=record["bev"],
            bev_cfg=cfg.bev,
            gt_boxes=gt_boxes,
            pred_boxes=pred_boxes,
            pred_scores=pred_scores,
            title=f"{args.split} sample={record['sample_index']}",
        )

        drive_id = int(box_record["drive_id"])
        frame_idx = int(box_record["frame_idx"])
        stem = f"drive_{drive_id:04d}_frame_{frame_idx:06d}"
        cv2.imwrite(str(bev_dir / f"{stem}.png"), cv2.cvtColor(bev_img, cv2.COLOR_RGB2BGR))

        if raw_record is not None:
            camera = np.asarray(raw_record["image_02"], dtype=np.uint8)
            points = np.asarray(raw_record["points"], dtype=np.float32)
            cam_img = _overlay_points(camera, points, proj_cam2)
            cam_img = _draw_projected_boxes(cam_img, gt_boxes, proj_cam2, color=(0, 255, 0))
            cam_img = _draw_projected_boxes(cam_img, pred_boxes, proj_cam2, color=(255, 0, 0))
            cv2.imwrite(str(cam_dir / f"{stem}.png"), cv2.cvtColor(cam_img, cv2.COLOR_RGB2BGR))

        has_cam = raw_record is not None
        print(
            f"saved {stem}: gt_boxes={gt_boxes.shape[0]} pred_boxes={pred_boxes.shape[0]} "
            f"score_thresh={cfg.infer.score_threshold:.4f} cam={'yes' if has_cam else 'skipped'}"
        )

    print(f"saved_viz={out_dir}")


if __name__ == "__main__":
    main()
