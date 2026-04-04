from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from lidar_det.config import BEVConfig
from lidar_det.data.bev import bev_tensor_to_rgb
from lidar_det.eval.iou import box_to_corners_bev


def _metric_xy_to_pixel(
    x: np.ndarray,
    y: np.ndarray,
    cfg: BEVConfig,
    map_shape: Optional[Tuple[int, int]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    if map_shape is None:
        map_h, map_w = cfg.grid_size
    else:
        map_h, map_w = map_shape
    x_res = (cfg.x_max - cfg.x_min) / float(map_w)
    y_res = (cfg.y_max - cfg.y_min) / float(map_h)
    col = (x - cfg.x_min) / x_res
    row = (map_h - 1) - ((y - cfg.y_min) / y_res)
    return col, row


def _plot_boxes(
    ax: plt.Axes,
    boxes: np.ndarray,
    cfg: BEVConfig,
    color: str,
    label: str,
    scores: Optional[np.ndarray] = None,
    map_shape: Optional[Tuple[int, int]] = None,
) -> None:
    if boxes.shape[0] == 0:
        return

    for i, box in enumerate(boxes):
        corners = box_to_corners_bev(box)
        xpix, ypix = _metric_xy_to_pixel(corners[:, 0], corners[:, 1], cfg, map_shape=map_shape)
        xpix = np.concatenate([xpix, xpix[:1]])
        ypix = np.concatenate([ypix, ypix[:1]])
        ax.plot(
            xpix,
            ypix,
            color=color,
            linewidth=1.8,
            antialiased=True,
            solid_capstyle="round",
            solid_joinstyle="round",
        )

        cx, cy = box[0], box[1]
        heading_len = box[3] * 0.5
        hx = cx + heading_len * np.cos(box[6])
        hy = cy + heading_len * np.sin(box[6])
        cpx, cpy = _metric_xy_to_pixel(
            np.array([cx, hx]), np.array([cy, hy]), cfg, map_shape=map_shape
        )
        ax.plot(
            cpx,
            cpy,
            color=color,
            linewidth=1.2,
            antialiased=True,
            solid_capstyle="round",
            solid_joinstyle="round",
        )

        if scores is not None:
            text_x, text_y = _metric_xy_to_pixel(
                np.array([cx]), np.array([cy]), cfg, map_shape=map_shape
            )
            ax.text(text_x[0], text_y[0], f"{scores[i]:.2f}", color=color, fontsize=7)

    ax.plot([], [], color=color, label=label)


def _style_axes(ax: plt.Axes, dark_mode: bool) -> None:
    if not dark_mode:
        return
    ax.set_facecolor("black")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")


def _style_legend(ax: plt.Axes, dark_mode: bool) -> None:
    if not dark_mode:
        return
    leg = ax.get_legend()
    if leg is None:
        return
    frame = leg.get_frame()
    frame.set_facecolor("black")
    frame.set_edgecolor("white")
    for txt in leg.get_texts():
        txt.set_color("white")


def plot_bev_with_boxes(
    bev: np.ndarray,
    bev_cfg: BEVConfig,
    gt_boxes: np.ndarray,
    pred_boxes: Optional[np.ndarray] = None,
    pred_scores: Optional[np.ndarray] = None,
    title: str = "",
    save_path: Optional[str] = None,
    show: bool = False,
    gt_color: str = "lime",
    pred_color: str = "red",
    base_image: Optional[np.ndarray] = None,
    dark_mode: bool = True,
) -> None:
    if base_image is not None:
        img = base_image
    else:
        img = bev_tensor_to_rgb(bev)

    fig, ax = plt.subplots(
        figsize=(10, 8),
        facecolor="black",
    )
    ax.imshow(img, interpolation="lanczos")
    _style_axes(ax, dark_mode=True)

    _plot_boxes(ax, gt_boxes, bev_cfg, color=gt_color, label="GT", map_shape=img.shape[:2])
    if pred_boxes is not None:
        _plot_boxes(
            ax,
            pred_boxes,
            bev_cfg,
            color=pred_color,
            label="Pred",
            scores=pred_scores,
            map_shape=img.shape[:2],
        )

    text_color = "white"
    ax.set_title(title, color=text_color)
    ax.set_xlabel("x-forward", color=text_color)
    ax.set_ylabel("y-left", color=text_color)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="upper right")
        _style_legend(ax, dark_mode=True)
    ax.grid(False)
    ax.set_aspect("equal")

    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            out,
            dpi=150,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
    if show:
        plt.show()
    plt.close(fig)


def plot_scalar_map_with_boxes(
    scalar_map: np.ndarray,
    bev_cfg: BEVConfig,
    gt_boxes: np.ndarray,
    pred_boxes: Optional[np.ndarray] = None,
    pred_scores: Optional[np.ndarray] = None,
    title: str = "",
    cmap: str = "viridis",
    save_path: Optional[str] = None,
    show: bool = False,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    if scalar_map.ndim != 2:
        raise ValueError(f"Expected scalar_map shape (H,W), got {scalar_map.shape}")

    fig, ax = plt.subplots(figsize=(10, 8), facecolor="black")
    ax.set_facecolor("black")
    im = ax.imshow(scalar_map, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="lanczos")
    map_shape = scalar_map.shape

    _plot_boxes(ax, gt_boxes, bev_cfg, color="lime", label="GT", map_shape=map_shape)
    if pred_boxes is not None:
        _plot_boxes(
            ax,
            pred_boxes,
            bev_cfg,
            color="red",
            label="Pred",
            scores=pred_scores,
            map_shape=map_shape,
        )

    ax.set_title(title, color="white")
    ax.set_xlabel("x-forward", color="white")
    ax.set_ylabel("y-left", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="upper right")
        _style_legend(ax, dark_mode=True)
    ax.grid(False)
    ax.set_aspect("equal")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.get_yticklabels(), color="white")
    cbar.outline.set_edgecolor("white")

    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    if show:
        plt.show()
    plt.close(fig)


def render_bev_with_boxes(
    bev: np.ndarray,
    bev_cfg: BEVConfig,
    gt_boxes: np.ndarray,
    pred_boxes: Optional[np.ndarray] = None,
    pred_scores: Optional[np.ndarray] = None,
    title: str = "",
    base_image: Optional[np.ndarray] = None,
    gt_color: str = "lime",
    pred_color: str = "red",
    dark_mode: bool = True,
) -> np.ndarray:
    """Render BEV + boxes to an RGB image array for TensorBoard logging."""
    if base_image is not None:
        img = base_image
    else:
        img = bev_tensor_to_rgb(bev)
    fig, ax = plt.subplots(
        figsize=(10, 8),
        facecolor="black",
    )
    ax.imshow(img, interpolation="lanczos")
    _style_axes(ax, dark_mode=True)

    _plot_boxes(ax, gt_boxes, bev_cfg, color=gt_color, label="GT", map_shape=img.shape[:2])
    if pred_boxes is not None:
        _plot_boxes(
            ax,
            pred_boxes,
            bev_cfg,
            color=pred_color,
            label="Pred",
            scores=pred_scores,
            map_shape=img.shape[:2],
        )
    text_color = "white"
    ax.set_title(title, color=text_color)
    ax.set_xlabel("x-forward", color=text_color)
    ax.set_ylabel("y-left", color=text_color)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="upper right")
        _style_legend(ax, dark_mode=True)
    ax.grid(False)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.canvas.draw()

    rgba = np.asarray(fig.canvas.buffer_rgba())
    rgb = rgba[:, :, :3].copy()
    plt.close(fig)
    return rgb
