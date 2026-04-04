#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lidar_det.config import default_config
from lidar_det.data.bev import bev_per_channel_to_rgb, bev_tensor_to_rgb, rasterize_points_to_bev
from lidar_det.data.kitti_raw import KittiRawSequence
from lidar_det.viz.bev_plot import plot_bev_with_boxes
from lidar_det.viz.rerun_viz import (
    init_rerun,
    log_bev_boxes2d,
    log_bev_image,
    log_boxes3d_wireframes,
    log_points3d,
    rerun_available,
    set_frame,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read KITTI raw + tracklets and visualize GT alignment.")
    parser.add_argument("--root", default="data", help="KITTI raw root directory")
    parser.add_argument("--seq-date", default="2011_09_26")
    parser.add_argument("--drive", default="0005")
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-end", type=int, default=40)
    parser.add_argument("--frame-step", type=int, default=8)
    parser.add_argument("--rerun", action="store_true", help="Log to rerun if installed")
    parser.add_argument("--out-dir", default="outputs/plots/01_kitti_raw")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = default_config()
    cfg.data.root_dir = args.root
    cfg.data.seq_date = args.seq_date
    cfg.data.seq_drive = args.drive

    sequence = KittiRawSequence(cfg.data)

    start = max(0, args.frame_start)
    end = min(args.frame_end, sequence.num_frames)
    frame_indices = list(range(start, end, max(1, args.frame_step)))

    print(f"sequence={args.seq_date}_drive_{args.drive}")
    print(f"num_frames={sequence.num_frames}")
    print(f"selected_frames={len(frame_indices)} [{start}, {end})")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    use_rerun = args.rerun and rerun_available()
    if args.rerun and not use_rerun:
        print("rerun-sdk not available, falling back to matplotlib-only output.")

    if use_rerun:
        init_rerun("kitti_raw_read_viz", spawn=True)

    expected_bev_shape = (len(cfg.bev.channels), *cfg.bev.grid_size)

    for i, frame_idx in enumerate(frame_indices):
        points = sequence.get_pointcloud(frame_idx)
        gt_boxes, gt_classes, gt_names = sequence.get_boxes_and_classes(frame_idx, bev_cfg=cfg.bev)
        bev = rasterize_points_to_bev(points, cfg.bev)

        # --- BEV rasterization diagnostics (first frame only) ---
        if i == 0:
            print(f"\n--- BEV diagnostics (frame {frame_idx:04d}) ---")
            print(f"  shape: {bev.shape}  (expected {expected_bev_shape})")
            if bev.shape != expected_bev_shape:
                print(f"  WARNING: shape mismatch!")
            occ = np.any(bev > 0, axis=0)
            print(f"  occupancy: {occ.sum()}/{occ.size} cells ({100.0 * occ.sum() / occ.size:.1f}%)")
            for ch_idx, ch_name in enumerate(cfg.bev.channels):
                vals = bev[ch_idx]
                nz = vals[vals > 0]
                if nz.size > 0:
                    print(f"  {ch_name:14s}: range=[{nz.min():.3f}, {nz.max():.3f}]  mean={nz.mean():.3f}")
                else:
                    print(f"  {ch_name:14s}: all zeros — check your implementation")
            print()

        print(
            f"frame={frame_idx:04d} points={points.shape[0]} "
            f"gt_boxes={gt_boxes.shape[0]} classes={sorted(set(gt_names)) if gt_names else []}"
        )

        save_path = out_dir / f"frame_{frame_idx:04d}.png"
        plot_bev_with_boxes(
            bev=bev,
            bev_cfg=cfg.bev,
            gt_boxes=gt_boxes,
            pred_boxes=None,
            title=f"Frame {frame_idx}",
            save_path=str(save_path),
            show=False,
        )

        # Per-channel visualizations
        from PIL import Image

        channel_imgs = bev_per_channel_to_rgb(bev, channel_names=cfg.bev.channels)
        for ch_name, ch_rgb in channel_imgs.items():
            ch_path = out_dir / f"frame_{frame_idx:04d}_{ch_name}.png"
            Image.fromarray(ch_rgb).save(ch_path)

        if use_rerun:
            set_frame(frame_idx)
            log_points3d("world/points", points)
            log_boxes3d_wireframes("world/gt_boxes3d", gt_boxes, color=(0, 255, 0))
            log_bev_image("bev/image", bev_tensor_to_rgb(bev))
            log_bev_boxes2d("bev/gt_boxes2d", gt_boxes, cfg.bev, color=(0, 255, 0))

    print(f"saved_visualizations={out_dir}")


if __name__ == "__main__":
    main()
