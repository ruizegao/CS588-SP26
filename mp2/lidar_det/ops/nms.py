from __future__ import annotations

from typing import Tuple

import numpy as np

from lidar_det.eval.iou import iou_bev


def nms_bev(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
    max_detections: int = 100,
    rotated: bool = True,
) -> np.ndarray:
    """Standard greedy NMS in BEV. Returns kept indices."""
    if boxes.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)

    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0 and len(keep) < max_detections:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break

        remaining = order[1:]
        ious = np.array(
            [iou_bev(boxes[i], boxes[j], rotated=rotated) for j in remaining], dtype=np.float32
        )
        order = remaining[ious <= iou_threshold]

    return np.array(keep, dtype=np.int64)


def class_aware_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    iou_threshold: float,
    max_detections: int = 100,
    rotated: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if boxes.shape[0] == 0:
        return boxes, scores, classes

    final_indices = []
    unique_classes = np.unique(classes)
    for cls_id in unique_classes:
        cls_mask = classes == cls_id
        cls_indices = np.where(cls_mask)[0]
        cls_keep_local = nms_bev(
            boxes[cls_mask],
            scores[cls_mask],
            iou_threshold=iou_threshold,
            max_detections=max_detections,
            rotated=rotated,
        )
        cls_keep = cls_indices[cls_keep_local]
        final_indices.extend(cls_keep.tolist())

    if len(final_indices) == 0:
        return (
            np.zeros((0, boxes.shape[1]), dtype=boxes.dtype),
            np.zeros((0,), dtype=scores.dtype),
            np.zeros((0,), dtype=classes.dtype),
        )

    final_indices = np.array(final_indices, dtype=np.int64)
    order = scores[final_indices].argsort()[::-1]
    final_indices = final_indices[order][:max_detections]
    return boxes[final_indices], scores[final_indices], classes[final_indices]
