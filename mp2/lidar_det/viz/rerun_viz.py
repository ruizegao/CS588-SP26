from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from lidar_det.config import BEVConfig
from lidar_det.eval.iou import box_to_corners_bev

try:
    import rerun as rr
except Exception:  # pragma: no cover
    rr = None


def rerun_available() -> bool:
    return rr is not None


def init_rerun(app_id: str, spawn: bool = True) -> None:
    if rr is None:
        return
    rr.init(app_id, spawn=spawn)


def set_frame(frame_idx: int) -> None:
    if rr is None:
        return
    rr.set_time_sequence("frame", frame_idx)


def log_points3d(path: str, points: np.ndarray) -> None:
    if rr is None or points.shape[0] == 0:
        return
    try:
        rr.log(path, rr.Points3D(points[:, :3]))
    except Exception:
        pass


def log_bev_image(path: str, image: np.ndarray) -> None:
    if rr is None:
        return
    try:
        rr.log(path, rr.Image(image))
    except Exception:
        pass


def _metric_xy_to_pixel(x: np.ndarray, y: np.ndarray, cfg: BEVConfig) -> tuple[np.ndarray, np.ndarray]:
    h, _ = cfg.grid_size
    col = (x - cfg.x_min) / cfg.resolution
    row = (h - 1) - ((y - cfg.y_min) / cfg.resolution)
    return col, row


def log_bev_boxes2d(
    path: str,
    boxes: np.ndarray,
    cfg: BEVConfig,
    color: Tuple[int, int, int] = (255, 0, 0),
) -> None:
    if rr is None or boxes.shape[0] == 0:
        return

    strips = []
    for box in boxes:
        corners = box_to_corners_bev(box)
        xpix, ypix = _metric_xy_to_pixel(corners[:, 0], corners[:, 1], cfg)
        poly = np.stack([xpix, ypix], axis=1)
        poly = np.concatenate([poly, poly[:1]], axis=0)
        strips.append(poly.astype(np.float32))

    try:
        if hasattr(rr, "LineStrips2D"):
            rr.log(path, rr.LineStrips2D(strips, colors=[color]))
    except Exception:
        pass


def _box_to_3d_wireframe(box: np.ndarray) -> np.ndarray:
    x, y, z, l, w, h, yaw = box.tolist()
    dx = l * 0.5
    dy = w * 0.5
    dz = h * 0.5
    corners = np.array(
        [
            [dx, dy, dz],
            [dx, -dy, dz],
            [-dx, -dy, dz],
            [-dx, dy, dz],
            [dx, dy, -dz],
            [dx, -dy, -dz],
            [-dx, -dy, -dz],
            [-dx, dy, -dz],
        ],
        dtype=np.float32,
    )
    c = np.cos(yaw)
    s = np.sin(yaw)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    corners = corners @ rot.T
    corners += np.array([x, y, z], dtype=np.float32)

    edges = [
        [0, 1, 2, 3, 0],
        [4, 5, 6, 7, 4],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
    strips = [corners[e] for e in edges]
    return np.array(strips, dtype=object)


def log_boxes3d_wireframes(
    path: str,
    boxes: np.ndarray,
    color: Tuple[int, int, int] = (255, 0, 0),
) -> None:
    if rr is None or boxes.shape[0] == 0:
        return
    strips = []
    for box in boxes:
        strips.extend(list(_box_to_3d_wireframe(box)))

    try:
        if hasattr(rr, "LineStrips3D"):
            rr.log(path, rr.LineStrips3D(strips, colors=[color]))
    except Exception:
        pass
