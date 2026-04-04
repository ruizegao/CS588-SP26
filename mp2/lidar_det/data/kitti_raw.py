from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET

import numpy as np
import torch
from torch.utils.data import Dataset

from lidar_det.config import BEVConfig, DataConfig, TargetConfig
from lidar_det.data.bev import rasterize_points_to_bev
from lidar_det.data.targets import encode_targets

try:
    import pykitti
except Exception:  # pragma: no cover
    pykitti = None


@dataclass
class ObjectLabel:
    class_name: str
    track_id: int
    frame_idx: int
    x: float
    y: float
    z: float
    l: float
    w: float
    h: float
    yaw: float
    occlusion: int = -1
    truncation: int = -1
    state: int = -1

    def as_box7(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z, self.l, self.w, self.h, self.yaw], dtype=np.float32)


def _parse_calib_file(path: Path) -> Dict[str, np.ndarray]:
    data: Dict[str, np.ndarray] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            vals = value.strip().split()
            if not vals:
                continue
            try:
                nums = np.array([float(v) for v in vals], dtype=np.float64)
            except ValueError:
                # Skip non-numeric entries (e.g. calib_time).
                continue
            data[key] = nums
    return data


class KittiRawSequence:
    def __init__(self, data_cfg: DataConfig):
        self.data_cfg = data_cfg
        self.root_dir = Path(data_cfg.root_dir)
        self.seq_dir = self.root_dir / data_cfg.seq_date / f"{data_cfg.seq_date}_drive_{data_cfg.seq_drive}_sync"
        self.velo_dir = self.seq_dir / "velodyne_points" / "data"
        self.tracklet_path = self.seq_dir / "tracklet_labels.xml"

        if not self.seq_dir.exists():
            raise FileNotFoundError(f"KITTI sequence directory not found: {self.seq_dir}")
        if not self.velo_dir.exists():
            raise FileNotFoundError(f"Velodyne directory not found: {self.velo_dir}")
        if not self.tracklet_path.exists():
            raise FileNotFoundError(f"Tracklet labels not found: {self.tracklet_path}")

        self._pykitti_raw = None
        if pykitti is not None:
            try:
                self._pykitti_raw = pykitti.raw(
                    str(self.root_dir), data_cfg.seq_date, data_cfg.seq_drive
                )
            except Exception:
                self._pykitti_raw = None

        self.velo_files = sorted(self.velo_dir.glob("*.bin"))
        self.num_frames = len(self.velo_files)
        if self.num_frames == 0:
            raise RuntimeError(f"No velodyne frames found under {self.velo_dir}")

        self.class_names = list(data_cfg.class_names)
        self.class_to_id = {name: i for i, name in enumerate(self.class_names)}

        self._velo_to_image_proj, self._image_shape = self._load_projection_to_cam2()
        self.frame_to_labels = self._parse_tracklets()

    def _canonicalize_class(self, object_type: str) -> Optional[str]:
        if object_type in self.class_to_id:
            return object_type
        if "Car" in self.class_to_id:
            if self.data_cfg.include_van_as_car and object_type == "Van":
                return "Car"
            if self.data_cfg.merge_all_vehicles_to_car:
                vehicle_classes = {"Car", "Van", "Truck", "Tram", "Bus", "Misc"}
                if object_type in vehicle_classes:
                    return "Car"
        return None

    def _load_projection_to_cam2(self) -> Tuple[np.ndarray, Tuple[int, int]]:
        date_dir = self.root_dir / self.data_cfg.seq_date
        cam_calib_path = date_dir / "calib_cam_to_cam.txt"
        velo_calib_path = date_dir / "calib_velo_to_cam.txt"

        cam_calib = _parse_calib_file(cam_calib_path)
        velo_calib = _parse_calib_file(velo_calib_path)

        tr = np.eye(4, dtype=np.float64)
        tr[:3, :3] = velo_calib["R"].reshape(3, 3)
        tr[:3, 3] = velo_calib["T"]

        r_rect = np.eye(4, dtype=np.float64)
        r_rect[:3, :3] = cam_calib["R_rect_00"].reshape(3, 3)

        p_rect_02 = cam_calib["P_rect_02"].reshape(3, 4)
        proj = p_rect_02 @ r_rect @ tr

        s_rect_02 = cam_calib["S_rect_02"]
        img_w = int(s_rect_02[0])
        img_h = int(s_rect_02[1])
        return proj, (img_h, img_w)

    def _parse_tracklets(self) -> Dict[int, List[ObjectLabel]]:
        frame_to_labels: Dict[int, List[ObjectLabel]] = {i: [] for i in range(self.num_frames)}

        xml_root = ET.parse(self.tracklet_path).getroot()
        tracklets_node = xml_root.find("tracklets")
        if tracklets_node is None:
            return frame_to_labels

        tracklet_items = tracklets_node.findall("item")
        for track_id, trk in enumerate(tracklet_items):
            object_type = (trk.findtext("objectType") or "").strip()
            canonical_cls = self._canonicalize_class(object_type)
            if canonical_cls is None:
                continue

            h = float(trk.findtext("h", default="0"))
            w = float(trk.findtext("w", default="0"))
            l = float(trk.findtext("l", default="0"))
            first_frame = int(trk.findtext("first_frame", default="0"))

            poses_node = trk.find("poses")
            if poses_node is None:
                continue

            pose_items = poses_node.findall("item")
            for local_idx, pose in enumerate(pose_items):
                frame_idx = first_frame + local_idx
                if frame_idx < 0 or frame_idx >= self.num_frames:
                    continue

                tx = float(pose.findtext("tx", default="0"))
                ty = float(pose.findtext("ty", default="0"))
                tz = float(pose.findtext("tz", default="0"))
                rz = float(pose.findtext("rz", default="0"))
                occlusion = int(pose.findtext("occlusion", default="-1"))
                truncation = int(pose.findtext("truncation", default="-1"))
                state = int(pose.findtext("state", default="-1"))

                # KITTI tracklets store tz at box bottom center; move to geometric center.
                z_center = tz + 0.5 * h

                label = ObjectLabel(
                    class_name=canonical_cls,
                    track_id=track_id,
                    frame_idx=frame_idx,
                    x=tx,
                    y=ty,
                    z=z_center,
                    l=l,
                    w=w,
                    h=h,
                    yaw=rz,
                    occlusion=occlusion,
                    truncation=truncation,
                    state=state,
                )
                frame_to_labels[frame_idx].append(label)

        return frame_to_labels

    def frame_indices(self, frame_start: int = 0, frame_end: Optional[int] = None) -> List[int]:
        end = self.num_frames if frame_end is None else min(frame_end, self.num_frames)
        start = max(frame_start, 0)
        return list(range(start, end))

    def get_pointcloud(self, frame_idx: int) -> np.ndarray:
        if frame_idx < 0 or frame_idx >= self.num_frames:
            raise IndexError(f"frame_idx={frame_idx} out of range [0, {self.num_frames})")

        if self._pykitti_raw is not None:
            try:
                points = self._pykitti_raw.get_velo(frame_idx)
                if points is not None and points.ndim == 2 and points.shape[1] >= 4:
                    return points.astype(np.float32)
            except Exception:
                pass

        velo_path = self.velo_files[frame_idx]
        points = np.fromfile(velo_path, dtype=np.float32).reshape(-1, 4)
        return points

    def _is_box_in_camera_fov(self, box: ObjectLabel) -> bool:
        pt = np.array([box.x, box.y, box.z, 1.0], dtype=np.float64)
        uvw = self._velo_to_image_proj @ pt
        depth = uvw[2]
        if depth <= 1e-6:
            return False

        u = uvw[0] / depth
        v = uvw[1] / depth
        img_h, img_w = self._image_shape
        return (0.0 <= u < img_w) and (0.0 <= v < img_h)

    @staticmethod
    def _is_box_in_bev_range(box: ObjectLabel, bev_cfg: BEVConfig) -> bool:
        return (
            bev_cfg.x_min <= box.x <= bev_cfg.x_max
            and bev_cfg.y_min <= box.y <= bev_cfg.y_max
        )

    def get_labels(
        self,
        frame_idx: int,
        bev_cfg: Optional[BEVConfig] = None,
        apply_camera_fov: Optional[bool] = None,
    ) -> List[ObjectLabel]:
        labels = self.frame_to_labels.get(frame_idx, [])
        out: List[ObjectLabel] = []
        use_fov = self.data_cfg.require_camera_fov if apply_camera_fov is None else apply_camera_fov

        for label in labels:
            # Keep only labeled/truncated-valid regions according to KITTI-like policy.
            if label.truncation >= 2:
                continue
            if bev_cfg is not None and not self._is_box_in_bev_range(label, bev_cfg):
                continue
            if use_fov and not self._is_box_in_camera_fov(label):
                continue
            out.append(label)

        return out

    def get_boxes_and_classes(
        self,
        frame_idx: int,
        bev_cfg: Optional[BEVConfig] = None,
        apply_camera_fov: Optional[bool] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        labels = self.get_labels(frame_idx, bev_cfg=bev_cfg, apply_camera_fov=apply_camera_fov)
        if len(labels) == 0:
            return (
                np.zeros((0, 7), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
                [],
            )

        boxes = np.stack([lbl.as_box7() for lbl in labels], axis=0).astype(np.float32)
        class_ids = np.array([self.class_to_id[lbl.class_name] for lbl in labels], dtype=np.int64)
        class_names = [lbl.class_name for lbl in labels]
        return boxes, class_ids, class_names


class KittiRawBEVDataset(Dataset):
    def __init__(
        self,
        sequence: KittiRawSequence,
        bev_cfg: BEVConfig,
        target_cfg: TargetConfig,
        frame_indices: Sequence[int],
    ):
        self.sequence = sequence
        self.bev_cfg = bev_cfg
        self.target_cfg = target_cfg
        self.frame_indices = list(frame_indices)
        self.num_classes = len(self.sequence.class_names)

    def __len__(self) -> int:
        return len(self.frame_indices)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        frame_idx = self.frame_indices[idx]
        points = self.sequence.get_pointcloud(frame_idx)
        boxes, class_ids, _ = self.sequence.get_boxes_and_classes(frame_idx, bev_cfg=self.bev_cfg)

        bev = rasterize_points_to_bev(points, self.bev_cfg)
        encoded = encode_targets(
            boxes=boxes,
            class_ids=class_ids,
            bev_cfg=self.bev_cfg,
            target_cfg=self.target_cfg,
            num_classes=self.num_classes,
        )

        targets_t = {
            "heatmap": torch.from_numpy(encoded["heatmap"]).float(),
            "inds": torch.from_numpy(encoded["inds"]).long(),
            "mask": torch.from_numpy(encoded["mask"]).bool(),
            "reg": torch.from_numpy(encoded["reg"]).float(),
            "height": torch.from_numpy(encoded["height"]).float(),
            "dims": torch.from_numpy(encoded["dims"]).float(),
            "rot": torch.from_numpy(encoded["rot"]).float(),
            "cls_ids": torch.from_numpy(encoded["cls_ids"]).long(),
        }

        return {
            "frame_idx": frame_idx,
            "bev": torch.from_numpy(bev).float(),
            "targets": targets_t,
            "gt_boxes": torch.from_numpy(boxes).float(),
            "gt_classes": torch.from_numpy(class_ids).long(),
            "points": points,
        }


def collate_kitti_raw_batch(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    bev = torch.stack([sample["bev"] for sample in batch], dim=0)

    target_keys = batch[0]["targets"].keys()
    targets = {}
    for key in target_keys:
        targets[key] = torch.stack([sample["targets"][key] for sample in batch], dim=0)

    frame_indices = torch.tensor([sample["frame_idx"] for sample in batch], dtype=torch.long)
    gt_boxes = [sample["gt_boxes"] for sample in batch]
    gt_classes = [sample["gt_classes"] for sample in batch]
    points = [sample["points"] for sample in batch]

    return {
        "frame_idx": frame_indices,
        "bev": bev,
        "targets": targets,
        "gt_boxes": gt_boxes,
        "gt_classes": gt_classes,
        "points": points,
    }


def describe_sequence(sequence: KittiRawSequence, frame_indices: Optional[Iterable[int]] = None) -> Dict[str, object]:
    frames = list(range(sequence.num_frames)) if frame_indices is None else list(frame_indices)
    counts = [len(sequence.get_labels(fi)) for fi in frames]
    return {
        "num_frames": sequence.num_frames,
        "num_selected_frames": len(frames),
        "gt_boxes_per_frame_mean": float(np.mean(counts) if counts else 0.0),
        "gt_boxes_per_frame_max": int(np.max(counts) if counts else 0),
        "gt_boxes_per_frame_min": int(np.min(counts) if counts else 0),
    }
