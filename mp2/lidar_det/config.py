from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple


@dataclass
class DataConfig:
    root_dir: str = "data"
    seq_date: str = "2011_09_26"
    seq_drive: str = "0005"
    class_names: List[str] = field(default_factory=lambda: ["Car"])
    include_van_as_car: bool = True
    merge_all_vehicles_to_car: bool = True
    require_camera_fov: bool = True


@dataclass
class BEVConfig:
    x_min: float = 0.0
    x_max: float = 70.4
    y_min: float = -40.0
    y_max: float = 40.0
    z_min: float = -3.0
    z_max: float = 1.0
    resolution: float = 0.2
    channels: List[str] = field(
        default_factory=lambda: ["max_height", "mean_height", "intensity", "density"]
    )
    out_stride: int = 4

    @property
    def grid_size(self) -> Tuple[int, int]:
        width = int(round((self.x_max - self.x_min) / self.resolution))
        height = int(round((self.y_max - self.y_min) / self.resolution))
        return height, width

    @property
    def output_grid_size(self) -> Tuple[int, int]:
        h, w = self.grid_size
        if h % self.out_stride != 0 or w % self.out_stride != 0:
            raise ValueError(
                f"BEV grid ({h}, {w}) must be divisible by out_stride={self.out_stride}"
            )
        return h // self.out_stride, w // self.out_stride

    @property
    def output_resolution(self) -> float:
        return self.resolution * self.out_stride


@dataclass
class TargetConfig:
    gaussian_overlap: float = 0.1
    min_gaussian_radius: int = 1
    max_objects: int = 128
    use_log_dims: bool = True


@dataclass
class ModelConfig:
    base_channels: int = 64
    use_batchnorm: bool = False
    extra_res_blocks: int = 0


@dataclass
class TrainConfig:
    seed: int = 42
    batch_size: int = 2
    num_workers: int = 0
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    num_steps: int = 3000
    log_every: int = 20
    save_every: int = 500
    checkpoint_dir: str = "outputs/checkpoints"
    resume_checkpoint: str = ""
    use_tensorboard: bool = True
    tensorboard_dir: str = "outputs/tensorboard"
    tb_log_every: int = 10
    tb_image_every: int = 100
    tb_dense_pred_heatmap_thresh: float = 0.01
    run_val: bool = True
    val_every: int = 100
    val_batch_size: int = 1

    heatmap_weight: float = 1.0
    offset_weight: float = 1.0
    height_weight: float = 1.0
    dims_weight: float = 1.0
    yaw_weight: float = 1.0

    frame_start: int = 0
    frame_end: int = 200
    overfit_num_frames: int = 50


@dataclass
class InferConfig:
    score_threshold: float = 0.2
    topk: int = 100
    nms_iou_threshold: float = 0.1
    max_detections: int = 100
    use_rotated_nms: bool = True


@dataclass
class EvalConfig:
    iou_threshold: float = 0.5


@dataclass
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    bev: BEVConfig = field(default_factory=BEVConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    infer: InferConfig = field(default_factory=InferConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    device: str = "cuda"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "AppConfig":
        cfg = AppConfig()
        for section_name in ["data", "bev", "target", "model", "train", "infer", "eval"]:
            section_dict = payload.get(section_name, {})
            section_obj = getattr(cfg, section_name)
            for key, value in section_dict.items():
                if hasattr(section_obj, key):
                    setattr(section_obj, key, value)
        if "device" in payload:
            cfg.device = payload["device"]
        return cfg


def default_config() -> AppConfig:
    return AppConfig()


def ensure_output_dirs(cfg: AppConfig) -> None:
    Path(cfg.train.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.train.tensorboard_dir).mkdir(parents=True, exist_ok=True)
    Path("outputs/plots").mkdir(parents=True, exist_ok=True)
    Path("outputs/eval").mkdir(parents=True, exist_ok=True)
