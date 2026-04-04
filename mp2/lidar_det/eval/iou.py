from __future__ import annotations

from typing import Iterable, List

import numpy as np


def box_to_corners_bev(box: np.ndarray) -> np.ndarray:
    """Convert [x, y, z, l, w, h, yaw] box to 4x2 BEV corners."""
    x, y, _, l, w, _, yaw = box.tolist()
    dx = l * 0.5
    dy = w * 0.5

    corners = np.array(
        [[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy]],
        dtype=np.float32,
    )
    c = np.cos(yaw)
    s = np.sin(yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    rotated = corners @ rot.T
    rotated[:, 0] += x
    rotated[:, 1] += y
    return rotated


def polygon_area(poly: np.ndarray) -> float:
    if poly.shape[0] < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return float(0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def polygon_signed_area(poly: np.ndarray) -> float:
    if poly.shape[0] < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return float(0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _inside(point: np.ndarray, edge_start: np.ndarray, edge_end: np.ndarray) -> bool:
    return (edge_end[0] - edge_start[0]) * (point[1] - edge_start[1]) - (
        edge_end[1] - edge_start[1]
    ) * (point[0] - edge_start[0]) >= 0


def _intersection(
    p1: np.ndarray, p2: np.ndarray, cp1: np.ndarray, cp2: np.ndarray
) -> np.ndarray:
    dc = cp1 - cp2
    dp = p1 - p2
    n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
    n2 = p1[0] * p2[1] - p1[1] * p2[0]
    denom = dc[0] * dp[1] - dc[1] * dp[0]
    if abs(denom) < 1e-8:
        return p2
    x = (n1 * dp[0] - n2 * dc[0]) / denom
    y = (n1 * dp[1] - n2 * dc[1]) / denom
    return np.array([x, y], dtype=np.float32)


def convex_polygon_intersection(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    output = subject.copy()
    clip_poly = clip.copy()

    # Sutherland-Hodgman assumes a consistent winding order.
    if polygon_signed_area(output) < 0:
        output = output[::-1]
    if polygon_signed_area(clip_poly) < 0:
        clip_poly = clip_poly[::-1]

    cp1 = clip[-1]

    cp1 = clip_poly[-1]
    for cp2 in clip_poly:
        input_list = output.copy()
        if input_list.shape[0] == 0:
            break
        output_list: List[np.ndarray] = []

        s = input_list[-1]
        for e in input_list:
            if _inside(e, cp1, cp2):
                if not _inside(s, cp1, cp2):
                    output_list.append(_intersection(s, e, cp1, cp2))
                output_list.append(e)
            elif _inside(s, cp1, cp2):
                output_list.append(_intersection(s, e, cp1, cp2))
            s = e

        if len(output_list) == 0:
            output = np.zeros((0, 2), dtype=np.float32)
        else:
            output = np.stack(output_list, axis=0).astype(np.float32)
        cp1 = cp2

    return output


def iou_bev_rotated(box1: np.ndarray, box2: np.ndarray) -> float:
    poly1 = box_to_corners_bev(box1)
    poly2 = box_to_corners_bev(box2)

    area1 = polygon_area(poly1)
    area2 = polygon_area(poly2)
    if area1 <= 1e-8 or area2 <= 1e-8:
        return 0.0

    inter_poly = convex_polygon_intersection(poly1, poly2)
    inter_area = polygon_area(inter_poly)
    union = area1 + area2 - inter_area
    if union <= 1e-8:
        return 0.0
    return float(inter_area / union)


def iou_bev_axis_aligned(box1: np.ndarray, box2: np.ndarray) -> float:
    x1, y1, _, l1, w1, _, _ = box1.tolist()
    x2, y2, _, l2, w2, _, _ = box2.tolist()

    x1_min, x1_max = x1 - l1 * 0.5, x1 + l1 * 0.5
    y1_min, y1_max = y1 - w1 * 0.5, y1 + w1 * 0.5
    x2_min, x2_max = x2 - l2 * 0.5, x2 + l2 * 0.5
    y2_min, y2_max = y2 - w2 * 0.5, y2 + w2 * 0.5

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    iw = max(0.0, inter_xmax - inter_xmin)
    ih = max(0.0, inter_ymax - inter_ymin)
    inter = iw * ih

    a1 = l1 * w1
    a2 = l2 * w2
    union = a1 + a2 - inter
    if union <= 1e-8:
        return 0.0
    return float(inter / union)


def iou_bev(box1: np.ndarray, box2: np.ndarray, rotated: bool = True) -> float:
    if rotated:
        return iou_bev_rotated(box1, box2)
    return iou_bev_axis_aligned(box1, box2)


def pairwise_iou_bev(boxes1: np.ndarray, boxes2: np.ndarray, rotated: bool = True) -> np.ndarray:
    n = boxes1.shape[0]
    m = boxes2.shape[0]
    out = np.zeros((n, m), dtype=np.float32)
    for i in range(n):
        for j in range(m):
            out[i, j] = iou_bev(boxes1[i], boxes2[j], rotated=rotated)
    return out
