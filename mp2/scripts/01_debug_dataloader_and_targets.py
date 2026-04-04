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
from lidar_det.data.kitti_raw import KittiRawBEVDataset, KittiRawSequence
from lidar_det.data.targets import decode_targets
from lidar_det.viz.bev_plot import plot_bev_with_boxes
from lidar_det.viz.rerun_viz import init_rerun, log_bev_boxes2d, log_bev_image, rerun_available, set_frame
from lidar_det.data.bev import bev_tensor_to_rgb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug dataloader + target encode/decode consistency.")
    parser.add_argument("--root", default="data")
    parser.add_argument("--seq-date", default="2011_09_26")
    parser.add_argument("--drive", default="0005")
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-end", type=int, default=40)
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--out-dir", default="outputs/plots/02_targets")
    return parser.parse_args()


def _mean_center_error(gt_boxes: np.ndarray, dec_boxes: np.ndarray) -> float:
    if gt_boxes.shape[0] == 0 and dec_boxes.shape[0] == 0:
        return 0.0
    if gt_boxes.shape[0] == 0 or dec_boxes.shape[0] == 0:
        return float("inf")

    gt_xy = gt_boxes[:, :2]
    dec_xy = dec_boxes[:, :2]

    dists = np.linalg.norm(gt_xy[:, None, :] - dec_xy[None, :, :], axis=2)
    used_gt = set()
    used_dec = set()
    pairs = []

    while len(used_gt) < gt_boxes.shape[0] and len(used_dec) < dec_boxes.shape[0]:
        best = None
        best_d = float("inf")
        for i in range(gt_boxes.shape[0]):
            if i in used_gt:
                continue
            for j in range(dec_boxes.shape[0]):
                if j in used_dec:
                    continue
                if dists[i, j] < best_d:
                    best_d = float(dists[i, j])
                    best = (i, j)
        if best is None:
            break
        used_gt.add(best[0])
        used_dec.add(best[1])
        pairs.append(best_d)

    if not pairs:
        return float("inf")
    return float(np.mean(pairs))


def main() -> None:
    args = parse_args()
    cfg = default_config()
    cfg.data.root_dir = args.root
    cfg.data.seq_date = args.seq_date
    cfg.data.seq_drive = args.drive

    sequence = KittiRawSequence(cfg.data)
    frame_indices = sequence.frame_indices(args.frame_start, args.frame_end)[: args.max_frames]

    dataset = KittiRawBEVDataset(
        sequence=sequence,
        bev_cfg=cfg.bev,
        target_cfg=cfg.target,
        frame_indices=frame_indices,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    use_rerun = args.rerun and rerun_available()
    if args.rerun and not use_rerun:
        print("rerun-sdk not available, proceeding with matplotlib outputs only.")
    if use_rerun:
        init_rerun("kitti_targets_debug", spawn=True)

    expected_bev_shape = (len(cfg.bev.channels), *cfg.bev.grid_size)
    expected_hm_shape = (1, *cfg.bev.output_grid_size)

    eps_center = []
    for i in range(len(dataset)):
        sample = dataset[i]
        frame_idx = int(sample["frame_idx"])
        bev = sample["bev"].numpy()
        gt_boxes = sample["gt_boxes"].numpy()

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

        # --- encode / decode round-trip ---
        encoded_np = {k: v.numpy() for k, v in sample["targets"].items()}

        if i == 0:
            hm = encoded_np["heatmap"]
            print(f"--- Encode diagnostics (frame {frame_idx:04d}) ---")
            print(f"  heatmap shape: {hm.shape}  (expected {expected_hm_shape})")
            print(f"  heatmap max:   {hm.max():.4f}  (expected 1.0 at object centers)")
            n_valid = int(encoded_np["mask"].sum())
            print(f"  encoded objects: {n_valid}  (gt boxes: {gt_boxes.shape[0]})")
            print()

        dec_boxes, dec_classes, _ = decode_targets(encoded_np, cfg.bev, cfg.target)

        center_err = _mean_center_error(gt_boxes, dec_boxes)
        eps_center.append(center_err)

        print(
            f"frame={frame_idx:04d} gt={gt_boxes.shape[0]} decoded={dec_boxes.shape[0]} "
            f"mean_center_err={center_err:.4f}m"
        )

        save_path = out_dir / f"frame_{frame_idx:04d}.png"
        plot_bev_with_boxes(
            bev=bev,
            bev_cfg=cfg.bev,
            gt_boxes=gt_boxes,
            pred_boxes=dec_boxes,
            pred_scores=None,
            title=f"Targets encode/decode: frame {frame_idx}",
            save_path=str(save_path),
            show=False,
        )

        if use_rerun:
            set_frame(frame_idx)
            log_bev_image("bev/image", bev_tensor_to_rgb(bev))
            log_bev_boxes2d("bev/gt", gt_boxes, cfg.bev, color=(0, 255, 0))
            log_bev_boxes2d("bev/decoded", dec_boxes, cfg.bev, color=(255, 0, 0))

    if eps_center:
        finite = [e for e in eps_center if np.isfinite(e)]
        mean_eps = float(np.mean(finite)) if finite else float("inf")
        print(f"\n--- Summary ---")
        print(f"overall_mean_center_error={mean_eps:.4f}m")
        if mean_eps < 0.5:
            print("PASS: encode/decode round-trip is consistent (error < 0.5m)")
        else:
            print("FAIL: large round-trip error — check encode_targets and decode_targets")

    print(f"saved_visualizations={out_dir}")
    print("Check the images: green=GT boxes, red=decoded boxes. They should overlap closely.")


if __name__ == "__main__":
    main()
