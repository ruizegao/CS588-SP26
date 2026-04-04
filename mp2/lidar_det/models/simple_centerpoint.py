from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from lidar_det.config import TrainConfig


class ResidualBlock(nn.Module):
    """A small residual block with optional projection on the skip path."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, use_batchnorm: bool = False):
        super().__init__()
        norm1 = nn.BatchNorm2d(out_ch) if use_batchnorm else nn.Identity()
        norm2 = nn.BatchNorm2d(out_ch) if use_batchnorm else nn.Identity()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = norm1
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = norm2
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_ch != out_ch:
            skip_layers = [nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False)]
            if use_batchnorm:
                skip_layers.append(nn.BatchNorm2d(out_ch))
            self.skip = nn.Sequential(*skip_layers)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = out + identity
        return self.relu(out)


class Head(nn.Module):
    """A lightweight 2-layer convolutional prediction head.

    Architecture: ``3×3 Conv → ReLU → 1×1 Conv``.  Each of the five output
    quantities (heatmap, reg, height, dims, rot) gets its own ``Head`` instance.
    These are already created for you in ``SimpleCenterPoint.__init__`` — you do
    **not** need to instantiate them yourself.
    """

    def __init__(self, in_ch: int, out_ch: int, hidden_ch: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, out_ch, kernel_size=1, padding=0, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleCenterPoint(nn.Module):
    """
    TODO(Task 2): Small BEV CNN with CenterPoint-style dense heads at stride 4.

    You must implement both __init__ and forward.

    The backbone layers are (use base_ch=64):

        Layer name      Block type       in_ch -> out_ch    stride
        ----------      ----------       ---------------    ------
        stem          : nn.Sequential    in_channels -> 64  1      (3x3 conv [+ BN] + ReLU)
        stem_res      : ResidualBlock    64 -> 64           1
        neck[0]       : ResidualBlock    64 -> 64           1
        down1         : ResidualBlock    64 -> 128          2
        mid1          : ResidualBlock    128 -> 128         1
        neck[1]       : ResidualBlock    128 -> 128         1
        down2         : ResidualBlock    128 -> 128         2
        mid2          : ResidualBlock    128 -> 128         1
        neck[2]       : ResidualBlock    128 -> 128         1
        extra_blocks  : nn.Sequential of extra_res_blocks ResidualBlocks (128->128, stride 1)

    The neck should be an nn.Sequential of 3 ResidualBlocks (neck[0], neck[1], neck[2]).
    Input  (B, 4, 400, 352) -> feature map (B, 128, 100, 88).

    Five prediction heads (already created for you below) each take the 128-ch feature
    map and produce: heatmap (B,K,...), reg (B,2,...), height (B,1,...), dims (B,3,...),
    rot (B,2,...).
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        base_ch: int = 64,
        use_batchnorm: bool = False,
        extra_res_blocks: int = 0,
    ):
        super().__init__()
        self._base_ch = base_ch
        # ======= STUDENT TODO __init__ START =======
        # TODO(student): create the backbone layers listed in the docstring above.
        # Refer to the architecture diagram in the handout for block order,
        # channel sizes, and strides. Pass use_batchnorm to every ResidualBlock.

        pass  # placeholder — replace with your layer definitions
        # ======= STUDENT TODO __init__ END =======

        feat_ch = base_ch * 2
        self.head_heatmap = Head(feat_ch, num_classes, hidden_ch=base_ch)
        self.head_reg = Head(feat_ch, 2, hidden_ch=base_ch)
        self.head_height = Head(feat_ch, 1, hidden_ch=base_ch)
        self.head_dims = Head(feat_ch, 3, hidden_ch=base_ch)
        self.head_rot = Head(feat_ch, 2, hidden_ch=base_ch)

        self._init_heads()

    def _init_heads(self) -> None:
        # Low initial heatmap scores improve focal loss stability.
        heatmap_last = self.head_heatmap.net[-1]
        if isinstance(heatmap_last, nn.Conv2d):
            nn.init.constant_(heatmap_last.bias, -2.19)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # ======= STUDENT TODO forward START =======
        # TODO(student): apply the backbone layers in order (see docstring / handout figure),
        # then pass the feature map through the five prediction heads.
        # Result before heads must be (B, 128, 100, 88).

        # placeholder — returns zeros with the correct output shape
        B = x.shape[0]
        feat_ch = self._base_ch * 2
        H_out = x.shape[2] // 4
        W_out = x.shape[3] // 4
        x = torch.zeros(B, feat_ch, H_out, W_out, device=x.device, dtype=x.dtype)
        # ======= STUDENT TODO forward END =======

        return {
            "heatmap": self.head_heatmap(x),
            "reg": self.head_reg(x),
            "height": self.head_height(x),
            "dims": self.head_dims(x),
            "rot": self.head_rot(x),
        }


def _sigmoid_clamped(x: torch.Tensor) -> torch.Tensor:
    """Apply sigmoid activation with numerical-stability clamping.

    Clamps output to ``(1e-4, 1 - 1e-4)`` so that subsequent ``log(p)`` and
    ``log(1-p)`` calls in the focal loss never hit ``-inf``.

    Use this on ``preds["heatmap"]`` before computing the focal loss in
    ``compute_losses``.

    Args:
        x: Tensor of any shape (typically ``(B, K, H, W)`` raw heatmap logits).

    Returns:
        Tensor of the same shape with values in ``(1e-4, 1 - 1e-4)``.
    """
    return torch.sigmoid(x).clamp(1e-4, 1 - 1e-4)


def _focal_loss_centerpoint(pred_hm: torch.Tensor, gt_hm: torch.Tensor) -> torch.Tensor:
    """
    TODO(Task 2): CenterNet-style focal loss for dense heatmap supervision.

    Inputs:
        pred_hm: sigmoid-clamped heatmap predictions, shape (B, K, H, W), values in (0,1).
        gt_hm:   dense Gaussian target heatmap, shape (B, K, H, W), values in [0,1].

    Returns:
        Scalar loss tensor.

    Formula (see handout "Loss Function"):
        pos pixels (gt == 1): -(1-p)^2 * log(p)
        neg pixels (gt <  1): -(1-gt)^4 * p^2 * log(1-p)
        Normalise the sum by N_pos.  If N_pos < 1, return only the neg term.
    """
    # ======= STUDENT TODO START (edit only inside this block) =======
    # TODO(student): implement the CenterNet focal loss

    # placeholder
    loss = torch.tensor(0.0, device=pred_hm.device, requires_grad=True)
    # ======= STUDENT TODO END (do not change code outside this block) =======

    return loss


def _gather_feat(feat: torch.Tensor, ind: torch.Tensor) -> torch.Tensor:
    """Gather feature vectors at specified indices along dimension 1.

    Low-level helper used by :func:`_transpose_and_gather_feat`.  You generally
    do not need to call this directly.

    Args:
        feat: ``(B, N, C)`` tensor — flattened feature array.
        ind:  ``(B, k)`` long tensor — indices into the *N* dimension.

    Returns:
        ``(B, k, C)`` tensor — selected feature vectors.
    """
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    return feat.gather(1, ind)


def _transpose_and_gather_feat(feat: torch.Tensor, ind: torch.Tensor) -> torch.Tensor:
    """Gather from a dense ``(B, C, H, W)`` prediction map at flat spatial indices.

    Converts a dense head output to sparse per-object features by permuting to
    ``(B, H*W, C)`` and indexing.  Use this inside ``_reg_l1_loss`` to extract
    predicted values at the ground-truth object center locations.

    Args:
        feat: ``(B, C, H, W)`` tensor — dense prediction map from a model head.
        ind:  ``(B, max_objs)`` long tensor — flat spatial indices
            (``v * W + u``), each in ``[0, H*W)``.

    Returns:
        ``(B, max_objs, C)`` tensor — gathered predictions at object centers.

    Example::

        >>> pred = _transpose_and_gather_feat(pred_map, inds)  # (B, max_objs, C)
    """
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def _reg_l1_loss(
    pred_map: torch.Tensor, target: torch.Tensor, inds: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    TODO(Task 2): Masked L1 loss for sparse regression supervision.

    Inputs:
        pred_map (B, C, H, W): dense prediction map.
        target   (B, max_objs, C): sparse regression targets at object centers.
        inds     (B, max_objs):    flat grid index of each object center.
        mask     (B, max_objs):    1 where a valid object exists, 0 otherwise.

    Returns:
        Scalar loss: mean L1 error over masked (valid) object slots.
    """
    # ======= STUDENT TODO START (edit only inside this block) =======
    # TODO(student): implement the masked L1 regression loss
    #
    # Steps:
    #   1. Gather predictions at object centers:
    #        pred = _transpose_and_gather_feat(pred_map, inds)  -> (B, max_objs, C)
    #   2. Expand mask to (B, max_objs, 1) and cast to float.
    #   3. Compute: loss = sum(|pred * mask - target * mask|) / max(mask.sum(), 1)

    # placeholder
    loss = torch.tensor(0.0, device=pred_map.device, requires_grad=True)
    # ======= STUDENT TODO END (do not change code outside this block) =======

    return loss


def compute_losses(
    preds: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], train_cfg: TrainConfig
) -> Dict[str, torch.Tensor]:
    """
    TODO(Task 2): Assemble the weighted multi-task training loss.

    Inputs:
        preds:     dict of model head outputs (heatmap, reg, height, dims, rot).
        targets:   dict of supervision targets (heatmap, inds, mask, reg, height, dims, rot).
        train_cfg: loss weights (heatmap_weight, offset_weight, height_weight,
                   dims_weight, yaw_weight).

    Returns:
        Dictionary containing:
            "total":   weighted sum of all five loss terms.
            "heatmap": focal loss on the center heatmap.
            "reg":     masked L1 on the 2D sub-cell offset.
            "height":  masked L1 on the center z coordinate.
            "dims":    masked L1 on the box dimensions.
            "rot":     masked L1 on the [sin, cos] yaw encoding.
    """
    # ======= STUDENT TODO START (edit only inside this block) =======
    # TODO(student): compute all five loss terms and combine with weights
    #
    # Steps:
    #   1. Apply _sigmoid_clamped() to preds["heatmap"] before computing focal loss.
    #   2. Extract shared indices: inds = targets["inds"], mask = targets["mask"].
    #   3. Compute focal loss on heatmap, and masked L1 for reg, height, dims, rot.
    #   4. Weighted sum: train_cfg.heatmap_weight, offset_weight, height_weight,
    #      dims_weight, yaw_weight.

    # placeholders
    zero = torch.tensor(0.0, device=preds["heatmap"].device, requires_grad=True)
    loss_heatmap = zero
    loss_reg = zero
    loss_height = zero
    loss_dims = zero
    loss_rot = zero
    total = zero
    # ======= STUDENT TODO END (do not change code outside this block) =======

    return {
        "total": total,
        "heatmap": loss_heatmap,
        "reg": loss_reg,
        "height": loss_height,
        "dims": loss_dims,
        "rot": loss_rot,
    }


def train_step(
    model: nn.Module,
    bev: torch.Tensor,
    targets: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    train_cfg: TrainConfig,
) -> Dict[str, torch.Tensor]:
    """
    TODO(Task 2): Perform one training step — forward, loss, backward, update.

    This function implements the core training loop body: zero gradients,
    forward pass, loss computation, backward pass, and optimizer step.

    See the PyTorch training tutorial for background:
    https://docs.pytorch.org/tutorials/beginner/introyt/trainingyt.html

    Args:
        model: The CenterPoint model (already on the correct device).
        bev: ``(B, C, H, W)`` input BEV tensor (already on device).
        targets: Dict of target tensors (already on device).
        optimizer: The optimizer (Adam or AdamW).
        train_cfg: Training configuration (contains loss weights).

    Returns:
        The loss dictionary from ``compute_losses`` (keys: ``total``,
        ``heatmap``, ``reg``, ``height``, ``dims``, ``rot``).
    """
    # ======= STUDENT TODO START (edit only inside this block) =======
    # TODO(student): implement the training step
    #
    # Steps:
    #   1. Zero the optimizer gradients 
    #   2. Run the forward pass
    #   3. Compute losses
    #   4. Backward pass
    #   5. Update weights
    #   6. Return the loss dictionary.

    # placeholder — returns zero losses without updating the model
    zero = torch.tensor(0.0, device=bev.device, requires_grad=True)
    return {
        "total": zero,
        "heatmap": zero,
        "reg": zero,
        "height": zero,
        "dims": zero,
        "rot": zero,
    }
    # ======= STUDENT TODO END (do not change code outside this block) =======
