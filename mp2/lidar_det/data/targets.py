from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from lidar_det.config import BEVConfig, TargetConfig


def gaussian2d(shape: Tuple[int, int], sigma: float = 1.0) -> np.ndarray:
    """Create a 2D Gaussian kernel centered in a grid of the given shape.

    Used internally by :func:`draw_gaussian` to build the Gaussian blob that
    is stamped onto the heatmap at each object center.

    Args:
        shape: (height, width) of the output kernel array.
        sigma: Standard deviation of the Gaussian.  A common choice is
            ``diameter / 6`` so that values near the boundary are close to zero.

    Returns:
        np.ndarray of shape ``(height, width)`` with dtype float64.
        Peak value is 1.0 at the center; entries below machine-epsilon are
        clamped to 0.

    Example::

        >>> gaussian2d((5, 5), sigma=1.0).shape
        (5, 5)
    """
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def gaussian_radius(det_size: Tuple[float, float], min_overlap: float = 0.1) -> float:
    """Compute the smallest Gaussian radius that keeps a minimum IoU overlap.

    Given an object's footprint on the output grid, this function solves for the
    largest radius *r* such that a circle of that radius, centered on the object,
    still produces at least ``min_overlap`` IoU with the ground-truth box in
    three different overlap scenarios (inner, outer, and equal-size).  The
    returned value is the minimum across the three cases — i.e., the tightest
    (most conservative) radius.

    This is the standard CenterNet/CornerNet radius calculation.

    Args:
        det_size: ``(height_pixels, width_pixels)`` — object footprint in
            **output-grid pixels** (not metric).  Compute as
            ``(w_metric / output_resolution, l_metric / output_resolution)``.
        min_overlap: Minimum IoU between the Gaussian circle and the GT box.
            Default 0.1.

    Returns:
        Gaussian radius in output-grid pixels (float).  Typically rounded to
        ``int`` and clamped to ``target_cfg.min_gaussian_radius`` before use.

    Example::

        >>> radius = gaussian_radius((6.0, 3.0), min_overlap=0.1)
        >>> radius = max(min_gaussian_radius, int(radius))
    """
    height, width = det_size

    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(max(0.0, b1**2 - 4 * a1 * c1))
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(max(0.0, b2**2 - 4 * a2 * c2))
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(max(0.0, b3**2 - 4 * a3 * c3))
    r3 = (b3 + sq3) / 2 if a3 != 0 else 0

    return float(min(r1, r2, r3))


def draw_gaussian(heatmap: np.ndarray, center: Tuple[int, int], radius: int, k: float = 1.0) -> None:
    """Paint a Gaussian blob onto a single-class heatmap at the given center.

    The Gaussian is drawn with ``np.maximum`` so overlapping objects keep the
    highest value at each pixel (no destructive overwrite).  Boundary clipping
    is handled automatically — centers near the edge of the heatmap are safe.

    This function modifies ``heatmap`` **in-place** and returns nothing.

    Args:
        heatmap: 2-D array of shape ``(H_out, W_out)`` — one class channel
            of the full ``(K, H_out, W_out)`` heatmap.  Pass
            ``heatmap[cls_id]`` for the class of interest.
        center: ``(⌊u⌋, ⌊v⌋)`` — integer output-grid coordinates, given as
            **(column, row)** order.  ``⌊u⌋`` indexes into the ``W_out``
            axis and ``⌊v⌋`` indexes into the ``H_out`` axis.

            .. note::

                Inside the function, these are unpacked as ``x, y = center``
                where the local variable ``x`` = column = ``⌊u⌋`` and ``y`` =
                row = ``⌊v⌋``.  Do **not** confuse these with metric ``(x, y)``
                coordinates.
        radius: Gaussian radius in pixels, as returned by
            :func:`gaussian_radius` (after rounding / clamping).
        k: Peak amplitude multiplier (default 1.0).

    Example::

        >>> heatmap = np.zeros((100, 88), dtype=np.float32)   # (H_out, W_out)
        >>> u_int, v_int = 44, 50   # column, row on output grid
        >>> draw_gaussian(heatmap, center=(u_int, v_int), radius=3)
    """
    diameter = 2 * radius + 1
    gaussian = gaussian2d((diameter, diameter), sigma=diameter / 6)

    x, y = center
    height, width = heatmap.shape

    left, right = min(x, radius), min(width - x - 1, radius)
    top, bottom = min(y, radius), min(height - y - 1, radius)

    if min(left, right, top, bottom) < 0:
        return

    masked_heatmap = heatmap[y - top : y + bottom + 1, x - left : x + right + 1]
    masked_gaussian = gaussian[
        radius - top : radius + bottom + 1,
        radius - left : radius + right + 1,
    ]
    if masked_gaussian.shape != masked_heatmap.shape:
        return
    np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)


def metric_to_output_grid(
    x: np.ndarray, y: np.ndarray, bev_cfg: BEVConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert metric LiDAR coordinates to **output-grid** (stride-4) coordinates.

    The output grid is the spatial resolution at which the model's prediction
    heads operate (``H_out × W_out = 100 × 88``).  The formulas are (see
    Representation Contract in the handout):

    .. math::

        u = \\frac{x - x_{\\min}}{s \\Delta}, \\qquad
        v = (H_{\\text{out}} - 1) - \\frac{y - y_{\\min}}{s \\Delta}

    The coordinate convention is:

    * **u** increases with x (forward in LiDAR frame) — u = 0 at ``x_min``,
      u = ``W_out`` at ``x_max``.  Corresponds to the column axis.
    * **v** decreases with y — v = 0 at the top of the grid (largest valid y),
      v = ``H_out - 1`` at the bottom (``y_min``).  Corresponds to the row axis.

    The returned coordinates are **continuous** (float).  The integer cell is
    ``(⌊u⌋, ⌊v⌋)`` and the sub-cell residual is ``(u − ⌊u⌋, v − ⌊v⌋)``,
    which is stored in the ``reg`` target.

    Args:
        x: ``(N,)`` float array — metric x coordinates (LiDAR frame).
        y: ``(N,)`` float array — metric y coordinates (LiDAR frame).
        bev_cfg: BEV configuration providing bounds and output resolution.

    Returns:
        u:     ``(N,)`` float32 — continuous column coordinates on the output grid.
        v:     ``(N,)`` float32 — continuous row coordinates on the output grid.
        valid: ``(N,)`` bool — True where ``(u, v)`` falls inside the grid bounds.

    Example::

        >>> u, v, valid = metric_to_output_grid(boxes[:, 0], boxes[:, 1], bev_cfg)
        >>> u_int, v_int = int(u[i]), int(v[i])        # integer cell (⌊u⌋, ⌊v⌋)
        >>> reg_offset   = [u[i] - u_int, v[i] - v_int]  # sub-cell residual
    """
    h_out, w_out = bev_cfg.output_grid_size
    res = bev_cfg.output_resolution

    u = (x - bev_cfg.x_min) / res
    y_idx = (y - bev_cfg.y_min) / res
    v = (h_out - 1) - y_idx
    valid = (u >= 0.0) & (u < w_out) & (v >= 0.0) & (v < h_out)
    return u.astype(np.float32), v.astype(np.float32), valid


def output_grid_to_metric(u: np.ndarray, v: np.ndarray, bev_cfg: BEVConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Convert output-grid coordinates back to metric LiDAR coordinates.

    Exact inverse of :func:`metric_to_output_grid`:

    .. math::

        x = x_{\\min} + u \\cdot s\\Delta, \\qquad
        y = y_{\\min} + (H_{\\text{out}} - 1 - v) \\cdot s\\Delta

    Use this in ``decode_targets`` to recover world-frame ``(x, y)`` from
    grid-space ``(u, v)`` after adding the sub-cell residual.

    Args:
        u: ``(N,)`` float array — column coordinates on the output grid.
            Typically the integer cell plus the sub-cell offset:
            ``u = xs + reg[:, 0]`` (where xs = ``inds % W_out``).
        v: ``(N,)`` float array — row coordinates on the output grid.
            ``v = ys + reg[:, 1]`` (where ys = ``inds // W_out``).
        bev_cfg: BEV configuration providing bounds and output resolution.

    Returns:
        x: ``(N,)`` float32 — metric x coordinates in the LiDAR frame.
        y: ``(N,)`` float32 — metric y coordinates in the LiDAR frame.
    """
    h_out, _ = bev_cfg.output_grid_size
    res = bev_cfg.output_resolution
    x = bev_cfg.x_min + u * res
    y = bev_cfg.y_min + ((h_out - 1) - v) * res
    return x.astype(np.float32), y.astype(np.float32)


def encode_targets(
    boxes: np.ndarray,
    class_ids: np.ndarray,
    bev_cfg: BEVConfig,
    target_cfg: TargetConfig,
    num_classes: int,
) -> Dict[str, np.ndarray]:
    """
    TODO(Task 1): Encode GT boxes into CenterPoint-style dense supervision maps.

    Inputs:
        boxes (N, 7): GT boxes [x, y, z, l, w, h, yaw] in metric LiDAR frame.
        class_ids (N,): Integer class id for each box.
        bev_cfg: BEV grid configuration.
        target_cfg: Target encoding configuration (gaussian overlap, log dims, etc.).
        num_classes: Number of semantic classes.

    Returns:
        Dictionary containing:
            "heatmap": (num_classes, H_out, W_out) — Gaussian center heatmap.
            "inds":    (max_objs,)  — flat output-grid index of each object center.
            "mask":    (max_objs,)  — 1 where a valid object was encoded, else 0.
            "reg":     (max_objs, 2) — sub-cell (u, v) offsets.
            "height":  (max_objs, 1) — metric z of each center.
            "dims":    (max_objs, 3) — [log(l), log(w), log(h)] or [l, w, h].
            "rot":     (max_objs, 2) — [sin(yaw), cos(yaw)].
            "cls_ids": (max_objs,)  — class id for each encoded object.
    """
    h_out, w_out = bev_cfg.output_grid_size
    max_objs = target_cfg.max_objects

    heatmap = np.zeros((num_classes, h_out, w_out), dtype=np.float32)
    inds = np.zeros((max_objs,), dtype=np.int64)
    mask = np.zeros((max_objs,), dtype=np.uint8)
    reg = np.zeros((max_objs, 2), dtype=np.float32)
    height = np.zeros((max_objs, 1), dtype=np.float32)
    dims = np.zeros((max_objs, 3), dtype=np.float32)
    rot = np.zeros((max_objs, 2), dtype=np.float32)
    cls_targets = np.zeros((max_objs,), dtype=np.int64)

    if boxes.size == 0:
        return {
            "heatmap": heatmap,
            "inds": inds,
            "mask": mask,
            "reg": reg,
            "height": height,
            "dims": dims,
            "rot": rot,
            "cls_ids": cls_targets,
        }

    # ======= STUDENT TODO START (edit only inside this block) =======
    # TODO(student): fill in heatmap, inds, mask, reg, height, dims, rot, cls_ids
    #
    # For each valid box (up to max_objs):
    #   1. Map center (x,y) to output-grid coords (u_f, v_f) via
    #      metric_to_output_grid(); skip if valid[i] is False.
    #   2. Compute Gaussian radius from the footprint in output pixels
    #      (res = bev_cfg.output_resolution):
    #        obj_w = l / res,  obj_h = w / res
    #        radius = max(target_cfg.min_gaussian_radius,
    #                     int(gaussian_radius((obj_h, obj_w), target_cfg.gaussian_overlap)))
    #      then paint heatmap[cls_id] at (u_i, v_i) with draw_gaussian().
    #   3. Record sparse regression targets at slot out_count:
    #        inds[out_count]      = v_i * w_out + u_i
    #        mask[out_count]      = 1
    #        reg[out_count]       = [u_f - u_i, v_f - v_i]
    #        height[out_count, 0] = boxes[i, 2]   (metric z)
    #        dims[out_count]      = log([l,w,h])  if use_log_dims else [l,w,h]
    #        rot[out_count]       = [sin(yaw), cos(yaw)]
    #        cls_targets[out_count] = cls_id

    # ======= STUDENT TODO END (do not change code outside this block) =======

    return {
        "heatmap": heatmap,
        "inds": inds,
        "mask": mask,
        "reg": reg,
        "height": height,
        "dims": dims,
        "rot": rot,
        "cls_ids": cls_targets,
    }


def decode_targets(
    encoded: Dict[str, np.ndarray], bev_cfg: BEVConfig, target_cfg: TargetConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    TODO(Task 1): Decode encoded GT targets back to metric boxes for sanity checking.

    Inputs:
        encoded: dictionary returned by encode_targets.
        bev_cfg: BEV grid configuration.
        target_cfg: Target encoding configuration.

    Returns:
        boxes   (N, 7): float32  [x, y, z, l, w, h, yaw] in metric LiDAR frame.
        classes (N,):   int64    class id for each box.
        scores  (N,):   float32  all 1.0 (GT detections).
    """
    # ======= STUDENT TODO START (edit only inside this block) =======
    # TODO(student): invert encode_targets to recover metric boxes
    #
    # Steps:
    #   1. Build a bool mask from encoded["mask"]; gather valid entries.
    #   2. Recover integer grid coords:
    #        ys = valid_inds // w_out,  xs = valid_inds % w_out
    #   3. Add sub-cell offsets:  u = xs + reg[:,0],  v = ys + reg[:,1]
    #   4. Convert (u,v) -> metric (x,y) with output_grid_to_metric().
    #   5. Recover dims: np.exp(dims) if use_log_dims, else dims as-is.
    #   6. Recover yaw: np.arctan2(rot[:,0], rot[:,1])
    #   7. Stack into (N,7) and return with class ids and unit scores.

    # placeholders
    boxes = np.zeros((0, 7), dtype=np.float32)
    classes = np.zeros((0,), dtype=np.int64)
    scores = np.zeros((0,), dtype=np.float32)
    # ======= STUDENT TODO END (do not change code outside this block) =======

    return boxes, classes, scores


def _nms_heatmap(heatmap: torch.Tensor, kernel: int = 3) -> torch.Tensor:
    """
    TODO(Task 3): Suppress non-local-maximum pixels in a heatmap (pseudo-NMS).

    For each pixel, apply a ``kernel × kernel`` max-pooling window.  If a
    pixel's value equals the pooled maximum it is a local peak — keep it.
    Otherwise zero it out.  This suppresses the Gaussian tails around each
    center so that only the peak pixel survives into the top-k selection.

    Use this inside ``decode_predictions`` (step 2), right after applying
    sigmoid + clamp to the raw heatmap logits.

    Args:
        heatmap: ``(B, K, H, W)`` tensor of heatmap scores (after sigmoid).
        kernel: Size of the max-pool window (default 3).

    Returns:
        ``(B, K, H, W)`` tensor with the same shape — only local maxima retain
        their original scores; all other pixels are set to 0.

    Hint:
        Use ``F.max_pool2d`` with ``stride=1`` and appropriate padding so the
        output has the same spatial size as the input.  Then compare the pooled
        result to the original to build a keep-mask.
    """
    # ======= STUDENT TODO START (edit only inside this block) =======
    # TODO(student): implement local-max suppression
    #
    # Steps:
    #   1. Compute padding so the output size equals the input size.
    #   2. Apply F.max_pool2d with stride=1 and that padding.
    #   3. Build a keep mask: pixels where pooled == original.
    #   4. Return heatmap * keep (zeros out non-maxima).

    # placeholder — passes heatmap through unchanged (no suppression)
    return heatmap
    # ======= STUDENT TODO END (do not change code outside this block) =======


def _topk(heatmap: torch.Tensor, k: int = 100):
    """Extract the top-*k* scoring detections from an NMS-suppressed heatmap.

    The function first picks the top-*k* scores **per class**, then selects the
    global top-*k* across all classes.  This gives you at most *k* candidate
    detections per batch item, along with their spatial locations and class IDs.

    Use this as **step 3** of ``decode_predictions``, after ``_nms_heatmap``.

    Args:
        heatmap: ``(B, K, H, W)`` tensor — NMS-suppressed heatmap scores.
        k: Maximum number of detections to keep per batch item (default 100).

    Returns:
        A tuple of five tensors, all of shape ``(B, k)``:

        - **topk_score** — confidence scores of the top-*k* detections.
        - **topk_inds**  — flat spatial indices into the ``H × W`` grid (use
          with :func:`_transpose_and_gather_feat` to gather regression values).
        - **topk_clses** — predicted class IDs (int), range ``[0, K-1]``.
        - **topk_ys**    — row (v) coordinates on the output grid (float).
        - **topk_xs**    — column (u) coordinates on the output grid (float).

    Example::

        >>> scores, inds, clses, ys, xs = _topk(hm_nms, k=100)
        >>> reg = _transpose_and_gather_feat(preds["reg"], inds)
        >>> xs  = xs + reg[..., 0]   # refine with sub-cell offset
        >>> ys  = ys + reg[..., 1]
    """
    b, c, h, w = heatmap.size()

    topk_scores, topk_inds = torch.topk(heatmap.view(b, c, -1), k)
    topk_inds = topk_inds % (h * w)
    topk_ys = (topk_inds // w).float()
    topk_xs = (topk_inds % w).float()

    topk_score, topk_ind = torch.topk(topk_scores.view(b, -1), k)
    topk_clses = (topk_ind // k).int()

    topk_inds = _gather_feat(topk_inds.view(b, -1, 1), topk_ind).view(b, k)
    topk_ys = _gather_feat(topk_ys.view(b, -1, 1), topk_ind).view(b, k)
    topk_xs = _gather_feat(topk_xs.view(b, -1, 1), topk_ind).view(b, k)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def _gather_feat(feat: torch.Tensor, ind: torch.Tensor) -> torch.Tensor:
    """Gather feature vectors at specified indices along dimension 1.

    This is a low-level indexing helper used by :func:`_topk` and
    :func:`_transpose_and_gather_feat`.  You generally do **not** need to call
    this directly — use ``_transpose_and_gather_feat`` instead.

    Args:
        feat: ``(B, N, C)`` tensor — a flattened feature array.
        ind:  ``(B, k)`` long tensor — indices into the *N* dimension.

    Returns:
        ``(B, k, C)`` tensor — the *C*-dimensional feature vectors at the
        selected indices.
    """
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    return feat


def _transpose_and_gather_feat(feat: torch.Tensor, ind: torch.Tensor) -> torch.Tensor:
    """Gather from a dense ``(B, C, H, W)`` map at flat spatial indices.

    Converts a dense prediction map into sparse per-detection features by:

    1. Permuting ``(B, C, H, W)`` → ``(B, H, W, C)`` and flattening spatial
       dims to get ``(B, H*W, C)``.
    2. Using ``_gather_feat`` to pick the *C*-dimensional vectors at the
       given flat indices.

    Use this to extract regression targets (reg, height, dims, rot) at the
    locations identified by :func:`_topk`.

    Args:
        feat: ``(B, C, H, W)`` tensor — a dense prediction map from one of
            the model heads (e.g., ``preds["reg"]``, ``preds["dims"]``).
        ind:  ``(B, k)`` long tensor — flat spatial indices (from ``_topk``),
            each in range ``[0, H*W)``.

    Returns:
        ``(B, k, C)`` tensor — the *C*-dimensional predictions gathered at
        each detection location.

    Example::

        >>> reg  = _transpose_and_gather_feat(preds["reg"],  inds)  # (B, k, 2)
        >>> dims = _transpose_and_gather_feat(preds["dims"], inds)  # (B, k, 3)
        >>> rot  = _transpose_and_gather_feat(preds["rot"],  inds)  # (B, k, 2)
    """
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def decode_predictions(
    preds: Dict[str, torch.Tensor],
    bev_cfg: BEVConfig,
    target_cfg: TargetConfig,
    score_threshold: float = 0.2,
    topk: int = 100,
) -> List[Dict[str, np.ndarray]]:
    """
    TODO(Task 3): Decode dense model head outputs into per-frame 3D box detections.

    Inputs:
        preds: dict with keys
            "heatmap" (B, K, H_out, W_out) — raw heatmap logits
            "reg"     (B, 2, H_out, W_out) — sub-cell offset predictions
            "height"  (B, 1, H_out, W_out) — center z predictions
            "dims"    (B, 3, H_out, W_out) — log-dimension (or raw) predictions
            "rot"     (B, 2, H_out, W_out) — [sin_yaw, cos_yaw] predictions
        bev_cfg: BEV grid configuration.
        target_cfg: Target encoding configuration.
        score_threshold: Minimum heatmap score to keep a candidate.
        topk: Maximum number of candidates per batch item before thresholding.

    Returns:
        List of B dicts, each with:
            "boxes"   (N, 7): float32  [x, y, z, l, w, h, yaw] in metric space.
            "scores"  (N,):   float32  confidence scores.
            "classes" (N,):   int64    predicted class ids.
    """
    # ======= STUDENT TODO START (edit only inside this block) =======
    # TODO(student): decode dense predictions into scored 3D boxes
    #
    # Pipeline (exact inverse of encode_targets):
    #   1. sigmoid + clamp heatmap logits to probabilities in (1e-4, 1-1e-4).
    #   2. Local-max suppression via _nms_heatmap() (your implementation above).
    #   3. Extract top-k candidates per batch item with _topk() (provided above),
    #      giving scores, flat inds, class ids, grid ys, grid xs.
    #   4. Gather regression values at those indices with
    #      _transpose_and_gather_feat() (provided above).
    #   5. Refine grid coords:  xs += reg[...,0],  ys += reg[...,1]
    #   6. Convert to metric, where res = bev_cfg.output_resolution
    #        x = bev_cfg.x_min + xs * res
    #        y = bev_cfg.y_min + ((h_out - 1) - ys) * res
    #   7. Recover dims: torch.exp(dims) if use_log_dims, else as-is.
    #   8. Recover yaw: torch.atan2(rot[...,0], rot[...,1])
    #   9. Filter by score_threshold; stack into (N,7) boxes per batch item.

    # placeholders
    bsz = list(preds.values())[0].shape[0]
    out = [
        {
            "boxes": np.zeros((0, 7), dtype=np.float32),
            "scores": np.zeros((0,), dtype=np.float32),
            "classes": np.zeros((0,), dtype=np.int64),
        }
        for _ in range(bsz)
    ]
    # ======= STUDENT TODO END (do not change code outside this block) =======

    return out
