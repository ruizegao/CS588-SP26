# CS588 MP2: Center-Based 3D Detection From LiDAR

## Goal

In this MP you will implement a small but complete CenterPoint-style detector for single-sweep Velodyne LiDAR on KITTI raw sequences with tracklets. You will learn and implement every steps of the pipelines, from data processing, model architecture, hyperparamters tuning, and evaluation as a true ML engineer working in Autonomous Driving. 

## Setup

```bash
git clone https://github.com/hungdche/CS588-SP26.git
cd CS588-SP26/mp2

conda create -n cs588 python=3.11.0
conda activate cs588
pip install -r requirements.txt
```

Make sure that your torch is installed correctly and in compliance with your cuda version. One quick way to test if working correctly is
```bash
python -c "import torch; torch.tensor([1]).cuda(); print('Pass')"
```
If the terminal print "Pass", then torch is installed correctly. 

## Data

Please refer to the Canvas announcement for the link to download the data. Please put the unzipped `data` folder into `mp2` as shown below.
```text
mp2
├── data
│   ├── 2011_09_26
│   │   ├── calib_cam_to_cam.txt
│   │   ├── calib_imu_to_velo.txt
│   │   ├── calib_velo_to_cam.txt
│   │   ├── 2011_09_26_drive_0005_sync
│   │   ├── 2011_09_26_drive_0057_sync
│   │   └── ...
│   └── processed
│       ├── train.npz
│       ├── val.npz
│       ├── test.npz
│       ├── minival.npz
│       └── metadata.json
```

**IMPORTANT**: Please do not distribute the data. 

## How To Run

### Task 0: Read Raw Data And Visualize GT

```bash
python scripts/00_read_and_viz_kitti_raw.py
```

### Task 1: Data Preparation Debug

```bash
python scripts/01_debug_dataloader_and_targets.py
```

### Task 2: Model + Loss Debug On Tiny Minival

```bash
python scripts/02_train_processed_detector.py \
  --train-split minival \
  --val-split minival \
  --test-split minival \
  --epochs 50 --batch-size 2 \
  --run-name task2_minival_debug
```


### Task 3: Inference Pipeline And Visualization

```bash
python scripts/03_infer_processed_viz.py \
  --checkpoint outputs/processed_train/task2_minival_debug/best_val.pt \
  --processed-dir data/processed \
  --raw-root data \
  --split minival \
  --max-debug-frames 10 \
  --f1-iou-thresh 0.5 \
  --out-dir outputs/processed_viz
```


### Task 4: Full Train/Val

```bash
python scripts/02_train_processed_detector.py \
  --processed-dir data/processed \
  --train-split train \
  --val-split val \
  --test-split test \
  --epochs 10 \
  --batch-size 8 \
  --val-batch-size 8 \
  --eval-iou-threshold 0.5 \
  --run-name task4_full_train
```


### Task 5: Hold-Out Evaluation

```bash
python scripts/05_eval_processed.py \
  --checkpoint outputs/processed_train/task4_full_train/best_val.pt \
  --processed-dir data/processed \
  --split test \
  --score-thresh 0.01 \
  --topk 100 \
  --nms-iou 0.1 \
  --iou-threshold 0.5 \
  --out-dir outputs/processed_eval/task4_full_train_test
```

## Notes

- LiDAR frame uses `x` forward, `y` left, `z` up.
- Box format is `[x, y, z, l, w, h, yaw]`.
- The one-class homework setup merges vehicle-like labels into `Car`.
