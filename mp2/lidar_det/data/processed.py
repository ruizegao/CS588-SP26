from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from lidar_det.config import TargetConfig


def _load_json(path: Path) -> Dict[str, object]:
    """Load a JSON file and return the parsed dictionary."""
    return json.loads(path.read_text(encoding="utf-8"))


def _load_np_allow_pickle(path: Path) -> np.ndarray:
    """Load a NumPy array allowing pickled object payloads."""
    return np.load(path, allow_pickle=True)


def _maybe_item(value: object) -> object:
    """Unwrap NumPy scalar objects into plain Python values when needed."""
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _dense_output_to_sparse_targets(
    output_whc: np.ndarray,
    num_classes: int,
    target_cfg: TargetConfig,
) -> Dict[str, np.ndarray]:
    """Convert a stored dense output tensor into sparse CenterPoint targets."""
    dense = np.transpose(output_whc.astype(np.float32), (2, 1, 0))
    heatmap = dense[:num_classes]
    offset_x = dense[num_classes + 0]
    offset_y = dense[num_classes + 1]
    center_z = dense[num_classes + 2]
    dims = dense[num_classes + 3 : num_classes + 6]
    sin_yaw = dense[num_classes + 6]
    cos_yaw = dense[num_classes + 7]
    reg_mask = dense[num_classes + 8] > 0.5

    h_out, w_out = reg_mask.shape
    ys, xs = np.nonzero(reg_mask)
    num_objs = min(len(xs), target_cfg.max_objects)

    inds = np.zeros((target_cfg.max_objects,), dtype=np.int64)
    mask = np.zeros((target_cfg.max_objects,), dtype=np.uint8)
    reg = np.zeros((target_cfg.max_objects, 2), dtype=np.float32)
    height = np.zeros((target_cfg.max_objects, 1), dtype=np.float32)
    dims_out = np.zeros((target_cfg.max_objects, 3), dtype=np.float32)
    rot = np.zeros((target_cfg.max_objects, 2), dtype=np.float32)
    cls_ids = np.zeros((target_cfg.max_objects,), dtype=np.int64)

    if num_objs == 0:
        return {
            "heatmap": heatmap.astype(np.float32),
            "inds": inds,
            "mask": mask,
            "reg": reg,
            "height": height,
            "dims": dims_out,
            "rot": rot,
            "cls_ids": cls_ids,
        }

    if len(xs) > target_cfg.max_objects:
        scores = heatmap[:, ys, xs].max(axis=0)
        keep = np.argsort(-scores)[: target_cfg.max_objects]
        ys = ys[keep]
        xs = xs[keep]

    num_objs = len(xs)
    inds[:num_objs] = ys.astype(np.int64) * w_out + xs.astype(np.int64)
    mask[:num_objs] = 1
    reg[:num_objs, 0] = offset_x[ys, xs]
    reg[:num_objs, 1] = offset_y[ys, xs]
    height[:num_objs, 0] = center_z[ys, xs]
    dims_out[:num_objs] = np.stack(
        [dims[0, ys, xs], dims[1, ys, xs], dims[2, ys, xs]],
        axis=1,
    ).astype(np.float32)
    rot[:num_objs, 0] = sin_yaw[ys, xs]
    rot[:num_objs, 1] = cos_yaw[ys, xs]
    cls_ids[:num_objs] = np.argmax(heatmap[:, ys, xs], axis=0).astype(np.int64)

    return {
        "heatmap": heatmap.astype(np.float32),
        "inds": inds,
        "mask": mask,
        "reg": reg,
        "height": height,
        "dims": dims_out,
        "rot": rot,
        "cls_ids": cls_ids,
    }


class ProcessedSplitStore:
    """Access one processed split with optional on-disk cache for fast reloads."""

    def __init__(
        self,
        processed_dir: str | Path,
        split_name: str,
        cache_dir: str | Path | None = None,
    ):
        self.processed_dir = Path(processed_dir)
        self.split_name = split_name
        self.metadata = _load_json(self.processed_dir / "metadata.json")
        self.cache_dir = Path(cache_dir) if cache_dir is not None else self.processed_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.split_meta = self._resolve_split_meta(split_name)
        self.sample_refs = self._resolve_sample_refs(split_name)
        self.num_classes = len(self.metadata["class_names"])
        self.target_cfg = TargetConfig(**self.metadata["target_config"])
        self.input_channels = list(self.metadata["input_channels"])
        self.output_channels = list(self.metadata["output_channels"])

        self._ensure_cache_files()
        split_cache_dir = self.cache_dir / split_name
        self.inputs = np.load(split_cache_dir / "inputs.npy", mmap_mode="r")
        self.outputs = np.load(split_cache_dir / "outputs.npy", mmap_mode="r")
        self.boxes = _load_np_allow_pickle(split_cache_dir / "boxes.npy")
        raw_path = split_cache_dir / "raw.npy"
        self.raw = _load_np_allow_pickle(raw_path) if raw_path.exists() else np.asarray([], dtype=object)

        self.raw_lookup = {}
        for raw_idx, payload in enumerate(self.raw):
            record = _maybe_item(payload)
            key = (int(record["drive_id"]), int(record["frame_idx"]))
            self.raw_lookup[key] = raw_idx

    def _resolve_split_meta(self, split_name: str) -> Dict[str, object]:
        """Return metadata for one split or the synthetic minival artifact."""
        if split_name in self.metadata["splits"]:
            return self.metadata["splits"][split_name]
        if split_name == "minival":
            minival_meta = dict(self.metadata["minival"])
            minival_meta["sample_refs"] = []
            return minival_meta
        raise KeyError(f"Unknown processed split: {split_name}")

    def _resolve_sample_refs(self, split_name: str) -> List[Dict[str, int]]:
        """Resolve canonical sample refs for a split, including minival indirection."""
        if split_name in self.metadata["splits"]:
            return list(self.metadata["splits"][split_name]["sample_refs"])
        if split_name == "minival":
            val_refs = list(self.metadata["splits"]["val"]["sample_refs"])
            source_indices = list(self.metadata["minival"]["source_indices"])
            return [dict(val_refs[idx]) for idx in source_indices]
        raise KeyError(f"Unknown processed split: {split_name}")

    def _ensure_cache_files(self) -> None:
        """Materialize cache `.npy` files for faster repeated access."""
        split_cache_dir = self.cache_dir / self.split_name
        required = [
            split_cache_dir / "inputs.npy",
            split_cache_dir / "outputs.npy",
            split_cache_dir / "boxes.npy",
        ]
        if all(path.exists() for path in required):
            return

        split_cache_dir.mkdir(parents=True, exist_ok=True)
        split_npz_path = Path(self.split_meta["path"])
        if not split_npz_path.is_absolute():
            split_npz_path = (Path.cwd() / split_npz_path).resolve()

        with np.load(split_npz_path, allow_pickle=True) as data:
            np.save(split_cache_dir / "inputs.npy", data["inputs"])
            np.save(split_cache_dir / "outputs.npy", data["outputs"])
            np.save(split_cache_dir / "boxes.npy", data["boxes"], allow_pickle=True)
            np.save(split_cache_dir / "raw.npy", data["raw"], allow_pickle=True)

    def get_box_record(self, sample_index: int) -> Dict[str, object]:
        """Return the stored annotation record for one sample."""
        return _maybe_item(self.boxes[sample_index])

    def get_raw_record(self, sample_index: int) -> Optional[Dict[str, object]]:
        """Return the stored raw debug record for one sample if available."""
        ref = self.sample_refs[sample_index]
        key = (int(ref["drive_id"]), int(ref["frame_idx"]))
        raw_idx = self.raw_lookup.get(key)
        if raw_idx is None:
            return None
        return _maybe_item(self.raw[raw_idx])

    def __len__(self) -> int:
        """Return the number of samples in this split."""
        return len(self.sample_refs)


class ProcessedBEVDataset(Dataset):
    """PyTorch dataset wrapper around a processed split artifact."""

    def __init__(
        self,
        store: ProcessedSplitStore,
        sample_indices: Optional[Sequence[int]] = None,
        include_targets: bool = True,
    ):
        self.store = store
        self.include_targets = include_targets
        if sample_indices is None:
            self.sample_indices = list(range(len(store)))
        else:
            self.sample_indices = [int(idx) for idx in sample_indices]

    def __len__(self) -> int:
        """Return the number of selected samples."""
        return len(self.sample_indices)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        """Return one processed sample ready for training or evaluation."""
        sample_index = self.sample_indices[idx]
        ref = self.store.sample_refs[sample_index]
        input_whc = self.store.inputs[sample_index]
        bev = np.transpose(input_whc.astype(np.float32), (2, 1, 0))
        if input_whc.dtype == np.uint8:
            bev /= 255.0

        box_record = self.store.get_box_record(sample_index)
        gt_boxes = np.asarray(box_record["boxes"], dtype=np.float32)
        gt_classes = np.asarray(box_record["class_ids"], dtype=np.int64)

        sample = {
            "frame_idx": int(sample_index),
            "sample_idx": int(sample_index),
            "drive_id": int(ref["drive_id"]),
            "raw_frame_idx": int(ref["frame_idx"]),
            "bev": torch.from_numpy(bev).float(),
            "gt_boxes": torch.from_numpy(gt_boxes).float(),
            "gt_classes": torch.from_numpy(gt_classes).long(),
        }

        if self.include_targets:
            targets_np = _dense_output_to_sparse_targets(
                self.store.outputs[sample_index],
                num_classes=self.store.num_classes,
                target_cfg=self.store.target_cfg,
            )
            sample["targets"] = {
                "heatmap": torch.from_numpy(targets_np["heatmap"]).float(),
                "inds": torch.from_numpy(targets_np["inds"]).long(),
                "mask": torch.from_numpy(targets_np["mask"]).bool(),
                "reg": torch.from_numpy(targets_np["reg"]).float(),
                "height": torch.from_numpy(targets_np["height"]).float(),
                "dims": torch.from_numpy(targets_np["dims"]).float(),
                "rot": torch.from_numpy(targets_np["rot"]).float(),
                "cls_ids": torch.from_numpy(targets_np["cls_ids"]).long(),
            }

        return sample


def collate_processed_batch(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Collate processed samples into the format expected by the training code."""
    out: Dict[str, object] = {
        "frame_idx": torch.tensor([sample["frame_idx"] for sample in batch], dtype=torch.long),
        "sample_idx": torch.tensor([sample["sample_idx"] for sample in batch], dtype=torch.long),
        "drive_id": torch.tensor([sample["drive_id"] for sample in batch], dtype=torch.long),
        "raw_frame_idx": torch.tensor([sample["raw_frame_idx"] for sample in batch], dtype=torch.long),
        "bev": torch.stack([sample["bev"] for sample in batch], dim=0),
        "gt_boxes": [sample["gt_boxes"] for sample in batch],
        "gt_classes": [sample["gt_classes"] for sample in batch],
    }

    if "targets" in batch[0]:
        target_keys = batch[0]["targets"].keys()
        out["targets"] = {
            key: torch.stack([sample["targets"][key] for sample in batch], dim=0)
            for key in target_keys
        }

    return out


__all__ = [
    "ProcessedSplitStore",
    "ProcessedBEVDataset",
    "collate_processed_batch",
]
