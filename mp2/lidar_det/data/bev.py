from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from lidar_det.config import BEVConfig


def metric_to_grid(x: np.ndarray, y: np.ndarray, cfg: BEVConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert metric LiDAR coordinates to BEV **input-grid** row/col indices.

    Maps world-frame ``(x, y)`` positions to integer ``(row, col)`` indices on
    the full-resolution BEV grid (``H × W = grid_size``).  The formulas are
    (see Representation Contract in the handout):

    .. math::

        c = \\lfloor (x - x_{\\min}) / \\Delta \\rfloor, \\qquad
        r = (H - 1) - \\lfloor (y - y_{\\min}) / \\Delta \\rfloor

    The coordinate convention is:

    * **col** increases with x (forward in LiDAR frame).
    * **row** decreases with y — **row 0** is the top of the image (largest
      valid y), **row H-1** is the bottom (smallest y, i.e., ``y_min``).

    Points outside the BEV bounds are flagged as invalid.

    Args:
        x: ``(N,)`` float array — metric x coordinates.
        y: ``(N,)`` float array — metric y coordinates.
        cfg: BEV configuration with ``x_min``, ``y_min``, ``resolution``,
            and ``grid_size``.

    Returns:
        row:   ``(N,)`` int64 — row indices (0 = top, largest y).
        col:   ``(N,)`` int64 — column indices (0 = left = ``x_min``).
        valid: ``(N,)`` bool  — True where ``(row, col)`` is inside the grid.

    Example::

        >>> row, col, valid = metric_to_grid(pts[:, 0], pts[:, 1], cfg)
        >>> flat_idx = row[valid] * W + col[valid]
    """
    h, w = cfg.grid_size
    col = np.floor((x - cfg.x_min) / cfg.resolution).astype(np.int64)
    y_bins = np.floor((y - cfg.y_min) / cfg.resolution).astype(np.int64)
    row = (h - 1) - y_bins
    valid = (col >= 0) & (col < w) & (row >= 0) & (row < h)
    return row, col, valid


def grid_to_metric(row: np.ndarray, col: np.ndarray, cfg: BEVConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Convert BEV input-grid ``(row, col)`` back to metric ``(x, y)``.

    Inverse of :func:`metric_to_grid`.  Returns the **cell center** in metric
    coordinates (adds 0.5 to the integer index before scaling):

    .. math::

        x = x_{\\min} + (c + 0.5) \\cdot \\Delta, \\qquad
        y = y_{\\min} + (H - 1 - r + 0.5) \\cdot \\Delta

    Args:
        row: ``(N,)`` int or float array — row indices on the BEV grid.
        col: ``(N,)`` int or float array — column indices on the BEV grid.
        cfg: BEV configuration.

    Returns:
        x: ``(N,)`` float32 — metric x at the cell center.
        y: ``(N,)`` float32 — metric y at the cell center.
    """
    h, _ = cfg.grid_size
    x = cfg.x_min + (col.astype(np.float32) + 0.5) * cfg.resolution
    y = cfg.y_min + ((h - 1 - row).astype(np.float32) + 0.5) * cfg.resolution
    return x, y


def rasterize_points_to_bev(points: np.ndarray, cfg: BEVConfig) -> np.ndarray:
    """
    TODO(Task 0): Rasterize a LiDAR point cloud into a 4-channel BEV tensor.

    Inputs:
        points (N, 4+): LiDAR points with columns [x, y, z, intensity, ...]
        cfg: BEV configuration (bounds, resolution, channel names)

    Returns:
        bev: float32 array of shape (C, H, W) = (len(cfg.channels), *cfg.grid_size)
             Channels in order: max_height, mean_height, intensity, density.
             Empty cells are 0 for all channels.
    """
    if points.ndim != 2 or points.shape[1] < 4:
        raise ValueError(f"Expected points shape (N,4+), got {points.shape}")

    # ======= STUDENT TODO START (edit only inside this block) =======
    # TODO(student): implement BEV rasterization
    #
    # Channel definitions (all normalized to [0,1]):
    #   max_height  : per-cell maximum z, normalized over [z_min, z_max]
    #   mean_height : per-cell mean z, normalized over [z_min, z_max]
    #   intensity   : per-cell mean intensity, clipped to [0,1]
    #   density     : log1p(count) / log1p(64), clipped to [0,1]
    #
    # Implementation hints:
    #   - Use metric_to_grid() to map (x,y) -> (row, col); also filter by z range.
    #   - Compute flat indices: flat_idx = row * W + col
    #   - Use np.bincount() for counts and weighted sums (fast vectorized reduce).
    #   - Use np.maximum.at() for per-cell max-height.
    #   - Do NOT loop over cells; use vectorized reductions.
    #   - max_height initializes to z_min (not 0) so np.maximum.at works.
    #   - After normalization, zero out ALL channels for empty cells
    #     (count == 0) using the per-cell count mask. Otherwise mean_height
    #     will be non-zero in empty cells due to the z_min offset.


    # placeholder
    c = len(cfg.channels)
    h, w = cfg.grid_size
    return np.zeros((c, h, w), dtype=np.float32)
    # ======= STUDENT TODO END (do not change code outside this block) =======


def bev_tensor_to_rgb(bev: np.ndarray) -> np.ndarray:
    """Convert CxHxW BEV tensor into uint8 HxWx3 image for visualization."""
    if bev.ndim != 3:
        raise ValueError(f"Expected CxHxW BEV, got {bev.shape}")
    import matplotlib.cm as mpl_cm

    c, h, w = bev.shape
    occ = np.max(bev, axis=0) > 0.0

    density_idx = 3 if c >= 4 else min(c - 1, 0)
    height_idx = 0
    intensity_idx = 2 if c >= 3 else min(c - 1, 0)

    density = np.clip(bev[density_idx], 0.0, 1.0)
    height = np.clip(bev[height_idx], 0.0, 1.0)
    intensity = np.clip(bev[intensity_idx], 0.0, 1.0)

    # Use a salient perceptual colormap for occupied cells while keeping empty
    # cells strictly black so box overlays remain easy to read.
    saliency = 0.60 * density + 0.25 * height + 0.15 * intensity
    saliency = np.clip(np.power(saliency, 0.85), 0.0, 1.0)
    rgba = mpl_cm.get_cmap("inferno")(saliency)
    rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
    rgb[~occ] = 0
    return rgb


def bev_per_channel_to_rgb(
    bev: np.ndarray,
    channel_names: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Convert each channel of a CxHxW BEV tensor into a separate HxWx3 uint8 image.

    Each channel is colormapped individually (viridis) with empty cells black.

    Args:
        bev: ``(C, H, W)`` float32 BEV tensor with values in ``[0, 1]``.
        channel_names: Optional list of length C giving a name per channel.
            If None, channels are named ``"ch0"``, ``"ch1"``, etc.

    Returns:
        Dict mapping channel name to ``(H, W, 3)`` uint8 RGB image.
    """
    import matplotlib.cm as mpl_cm

    if bev.ndim != 3:
        raise ValueError(f"Expected CxHxW BEV, got {bev.shape}")
    c, h, w = bev.shape
    if channel_names is None:
        channel_names = [f"ch{i}" for i in range(c)]

    occ = np.max(bev, axis=0) > 0.0
    out: dict[str, np.ndarray] = {}
    cmap = mpl_cm.get_cmap("viridis")
    for i, name in enumerate(channel_names):
        vals = np.clip(bev[i], 0.0, 1.0)
        rgba = cmap(vals)
        rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
        rgb[~occ] = 0
        out[name] = rgb
    return out
