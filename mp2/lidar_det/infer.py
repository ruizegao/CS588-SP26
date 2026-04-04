from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from lidar_det.config import AppConfig
from lidar_det.data.kitti_raw import KittiRawBEVDataset, KittiRawSequence, collate_kitti_raw_batch
from lidar_det.data.targets import decode_predictions
from lidar_det.models.simple_centerpoint import SimpleCenterPoint
from lidar_det.ops.nms import class_aware_nms
from lidar_det.train import resolve_device


FrameDetections = Dict[int, Dict[str, np.ndarray]]


def load_model_from_checkpoint(
    checkpoint_path: str,
    cfg: Optional[AppConfig] = None,
    device: str = "cpu",
) -> Tuple[SimpleCenterPoint, AppConfig]:
    ckpt = torch.load(checkpoint_path, map_location=device)

    if cfg is None:
        if "config" not in ckpt:
            raise ValueError("Checkpoint does not contain config and no cfg was provided.")
        cfg = AppConfig.from_dict(ckpt["config"])

    model = SimpleCenterPoint(
        in_channels=len(cfg.bev.channels),
        num_classes=len(cfg.data.class_names),
        base_ch=cfg.model.base_channels,
        use_batchnorm=cfg.model.use_batchnorm,
        extra_res_blocks=cfg.model.extra_res_blocks,
    )
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def build_infer_dataset(cfg: AppConfig, frame_indices: Optional[List[int]] = None):
    sequence = KittiRawSequence(cfg.data)
    if frame_indices is None:
        frame_indices = sequence.frame_indices(cfg.train.frame_start, cfg.train.frame_end)

    dataset = KittiRawBEVDataset(
        sequence=sequence,
        bev_cfg=cfg.bev,
        target_cfg=cfg.target,
        frame_indices=frame_indices,
    )
    return sequence, dataset


def run_inference(
    model: SimpleCenterPoint,
    cfg: AppConfig,
    dataset: KittiRawBEVDataset,
    batch_size: int = 2,
    device: Optional[torch.device] = None,
) -> Tuple[FrameDetections, FrameDetections]:
    if device is None:
        device = resolve_device(cfg.device)

    model = model.to(device)
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_kitti_raw_batch,
        drop_last=False,
    )

    pred_by_frame: FrameDetections = {}
    gt_by_frame: FrameDetections = {}

    with torch.no_grad():
        for batch in loader:
            bev = batch["bev"].to(device)
            frame_indices = batch["frame_idx"].cpu().numpy().tolist()

            outputs = model(bev)
            decoded = decode_predictions(
                outputs,
                bev_cfg=cfg.bev,
                target_cfg=cfg.target,
                score_threshold=cfg.infer.score_threshold,
                topk=cfg.infer.topk,
            )

            for b, frame_idx in enumerate(frame_indices):
                pred_boxes = decoded[b]["boxes"]
                pred_scores = decoded[b]["scores"]
                pred_classes = decoded[b]["classes"]

                pred_boxes, pred_scores, pred_classes = class_aware_nms(
                    pred_boxes,
                    pred_scores,
                    pred_classes,
                    iou_threshold=cfg.infer.nms_iou_threshold,
                    max_detections=cfg.infer.max_detections,
                    rotated=cfg.infer.use_rotated_nms,
                )

                pred_by_frame[int(frame_idx)] = {
                    "boxes": pred_boxes,
                    "scores": pred_scores,
                    "classes": pred_classes,
                }

                gt_boxes = batch["gt_boxes"][b].cpu().numpy().astype(np.float32)
                gt_classes = batch["gt_classes"][b].cpu().numpy().astype(np.int64)
                gt_by_frame[int(frame_idx)] = {
                    "boxes": gt_boxes,
                    "scores": np.ones((gt_boxes.shape[0],), dtype=np.float32),
                    "classes": gt_classes,
                }

    return pred_by_frame, gt_by_frame


__all__ = [
    "load_model_from_checkpoint",
    "build_infer_dataset",
    "run_inference",
]
