from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from lidar_det.eval.iou import iou_bev


FrameDetections = Dict[int, Dict[str, np.ndarray]]


def _compute_ap(rec: np.ndarray, prec: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap)


def evaluate_ap_per_class(
    predictions: FrameDetections,
    ground_truth: FrameDetections,
    class_id: int,
    iou_threshold: float = 0.5,
    rotated_iou: bool = True,
) -> Dict[str, np.ndarray | float | int]:
    det_list = []
    gt_count = 0

    gt_by_frame = {}
    for frame_idx, gt in ground_truth.items():
        boxes = gt.get("boxes", np.zeros((0, 7), dtype=np.float32))
        classes = gt.get("classes", np.zeros((0,), dtype=np.int64))
        cls_mask = classes == class_id
        gt_boxes = boxes[cls_mask]
        gt_by_frame[frame_idx] = {
            "boxes": gt_boxes,
            "matched": np.zeros((gt_boxes.shape[0],), dtype=bool),
        }
        gt_count += gt_boxes.shape[0]

    for frame_idx, pred in predictions.items():
        boxes = pred.get("boxes", np.zeros((0, 7), dtype=np.float32))
        scores = pred.get("scores", np.zeros((0,), dtype=np.float32))
        classes = pred.get("classes", np.zeros((0,), dtype=np.int64))

        cls_mask = classes == class_id
        boxes = boxes[cls_mask]
        scores = scores[cls_mask]

        for b, s in zip(boxes, scores):
            det_list.append((float(s), frame_idx, b.astype(np.float32)))

    if len(det_list) == 0:
        return {
            "ap": 0.0,
            "precision": np.zeros((0,), dtype=np.float32),
            "recall": np.zeros((0,), dtype=np.float32),
            "num_gt": gt_count,
            "num_det": 0,
        }

    det_list.sort(key=lambda x: x[0], reverse=True)

    tp = np.zeros((len(det_list),), dtype=np.float32)
    fp = np.zeros((len(det_list),), dtype=np.float32)

    for i, (_, frame_idx, det_box) in enumerate(det_list):
        gt_frame = gt_by_frame.get(frame_idx)
        if gt_frame is None or gt_frame["boxes"].shape[0] == 0:
            fp[i] = 1.0
            continue

        gt_boxes = gt_frame["boxes"]
        matched = gt_frame["matched"]

        best_iou = -1.0
        best_gt_idx = -1
        for j in range(gt_boxes.shape[0]):
            if matched[j]:
                continue
            iou = iou_bev(det_box, gt_boxes[j], rotated=rotated_iou)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp[i] = 1.0
            gt_frame["matched"][best_gt_idx] = True
        else:
            fp[i] = 1.0

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    recall = tp_cum / max(float(gt_count), 1e-6)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-6)
    ap = _compute_ap(recall, precision) if gt_count > 0 else 0.0

    return {
        "ap": ap,
        "precision": precision,
        "recall": recall,
        "num_gt": gt_count,
        "num_det": len(det_list),
    }


def evaluate_map(
    predictions: FrameDetections,
    ground_truth: FrameDetections,
    class_names: List[str],
    iou_threshold: float = 0.5,
    rotated_iou: bool = True,
) -> Dict[str, object]:
    per_class = {}
    ap_values = []

    for class_id, class_name in enumerate(class_names):
        cls_res = evaluate_ap_per_class(
            predictions=predictions,
            ground_truth=ground_truth,
            class_id=class_id,
            iou_threshold=iou_threshold,
            rotated_iou=rotated_iou,
        )
        per_class[class_name] = cls_res
        ap_values.append(float(cls_res["ap"]))

    mAP = float(np.mean(ap_values) if ap_values else 0.0)
    return {
        "mAP": mAP,
        "per_class": per_class,
        "iou_threshold": iou_threshold,
    }


def plot_pr_curves(eval_result: Dict[str, object], save_path: str) -> None:
    per_class = eval_result["per_class"]

    fig, ax = plt.subplots(figsize=(8, 6))
    plotted = False
    for class_name, res in per_class.items():
        recall = res["recall"]
        precision = res["precision"]
        ap = res["ap"]
        if len(recall) == 0:
            continue
        ax.plot(recall, precision, label=f"{class_name} AP={ap:.3f}")
        plotted = True

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"PR Curve (IoU={eval_result['iou_threshold']:.2f})")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True)
    if plotted:
        ax.legend(loc="lower left")

    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
