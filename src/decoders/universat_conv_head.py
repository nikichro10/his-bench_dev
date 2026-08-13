"""Faithful port of UniverSat's own segmentation/regression probe head.

https://github.com/gastruc/UniverSat: src/LP_eval_conv.py -- class ConvHead.
This is what UniverSat's own paper/repo used for its EnMAP (and other
hyperspectral) downstream evaluations, NOT the generic src.decoders.conv_head
already in this repo (that one does multiple internal upsample+conv steps to
reach full resolution; this one stays at token-grid resolution and lets the
caller's bilinear interpolate -- ConvSegmentor's own
_match_input_spatial_size -- do the upsampling, exactly matching their
LP_eval_conv.py's separate _upsample_logits_to_label_grid step).
"""

import torch.nn as nn
from torch import Tensor


class UniverSatConvHead(nn.Module):
    """Input: (B, D, ht, wt) token grid. Output: (B, C, ht, wt) -- same grid,
    no upsampling inside this module."""

    def __init__(
        self,
        num_input_features: int = 384,
        num_classes: int = 5,
    ):
        super().__init__()
        self.ln = nn.LayerNorm(num_input_features)
        # 3x3 conv with padding=1 preserves (ht, wt).
        self.head = nn.Sequential(
            nn.Conv2d(num_input_features, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)
        return self.head(x)
