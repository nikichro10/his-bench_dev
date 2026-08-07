# Ported from pangaea-bench_dev's pangaea/decoders/seg_mlp_fusion.py
# (MLPFusionDecoder), adapted to his-bench's decoupled backbone/decoder
# convention: pangaea's decoder owns the encoder and calls it internally
# (with its own freeze + multi-temporal L-TAE logic); here the decoder is a
# pure function of already-extracted feature maps, matching
# LightweightMultiTapSegHead's forward(features, output_size) contract.
# Encoder freezing is handled once, uniformly, by
# SemanticSegmentationModule.freeze_backbone -- not duplicated per-decoder.
# Multi-temporal / L-TAE fusion is dropped: his-bench has no multi-temporal
# datasets (single-timestep hyperspectral only), so that branch of the
# original would be dead code here.
#
# The fusion topology (per-tap 1x1 conv+ReLU projection -> concat -> 1x1
# conv+ReLU -> 3x3 conv+ReLU -> 1x1 classify) and all five upsample_mode
# options (bilinear / bicubic / nearest / transposed_conv / pixel_shuffle)
# are kept byte-for-byte identical to the pangaea original.

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

_ALIGN_CORNERS_MODES = {"bilinear", "bicubic"}


def _check_power_of_two(scale: int, mode: str) -> None:
    if scale <= 0 or (scale & (scale - 1)) != 0:
        raise ValueError(
            f"{mode} upsampling requires scale to be a power of 2, got {scale}. "
            f"Use bilinear or bicubic for non-power-of-2 patch sizes."
        )


def _build_transposed_upsample(num_classes: int, scale: int) -> nn.Sequential:
    _check_power_of_two(scale, "transposed_conv")
    n_steps = int(math.log2(scale))
    return nn.Sequential(*[
        nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1)
        for _ in range(n_steps)
    ])


class MLPFusionHead(nn.Module):
    """Multi-scale MLP fusion decoder for segmentation (pangaea-bench parity).

    Projects each backbone tap to `channels` with a 1x1 conv, concatenates
    them, fuses with a 1x1 + 3x3 conv pair, classifies with a final 1x1
    conv, and upsamples to the target resolution.

    Args:
        in_channels_list: per-tap channel counts (from the backbone probe,
            same source as LightweightMultiTapSegHead's in_channels_list).
        num_classes: number of segmentation classes.
        channels: projection width; concat gives N*channels input to fusion
            (default 128, matches pangaea's seg_mlp_fusion.yaml).
        upsample_mode: "bilinear" (default) / "bicubic" / "nearest" —
            F.interpolate after cls_head. "transposed_conv" — learned
            stride-2 ConvTranspose2d stack after cls_head; patch_size must
            be a power of 2. "pixel_shuffle" — replaces cls_head entirely
            with Conv(channels -> num_classes * r^2) + PixelShuffle(r)
            where r = patch_size; patch_size must be a power of 2.
        patch_size: required for "transposed_conv"/"pixel_shuffle" (the
            total spatial upsampling factor needed); auto-inferred from the
            backbone by infer_multitap_decoder_info the same way
            multiscale_conv_head's is, and passed in by
            SemanticSegmentationModule -- see is_mlp_fusion in
            semantic_segmentation_module.py.
    """

    def __init__(
        self,
        in_channels_list: Sequence[int],
        num_classes: int,
        channels: int = 128,
        upsample_mode: str = "bilinear",
        patch_size: int | None = None,
    ) -> None:
        super().__init__()

        if not in_channels_list:
            raise ValueError("in_channels_list must contain at least one tap.")
        if int(num_classes) <= 0:
            raise ValueError(f"num_classes must be > 0, got {num_classes}")

        self.in_channels_list = [int(c) for c in in_channels_list]
        self.num_classes = int(num_classes)
        self.channels = int(channels)
        self.upsample_mode = upsample_mode

        n_layers = len(self.in_channels_list)

        self.projections = nn.ModuleList([
            nn.Conv2d(dim, self.channels, kernel_size=1)
            for dim in self.in_channels_list
        ])

        self.fusion = nn.Sequential(
            nn.Conv2d(n_layers * self.channels, self.channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        if upsample_mode == "pixel_shuffle":
            if not patch_size:
                raise ValueError("pixel_shuffle upsample_mode requires patch_size.")
            r = int(patch_size)
            _check_power_of_two(r, "pixel_shuffle")
            # Replaces cls_head: one 1x1 conv to num_classes*r^2 channels, then PixelShuffle(r)
            self.cls_head = nn.Sequential(
                nn.Conv2d(self.channels, num_classes * r * r, kernel_size=1),
                nn.PixelShuffle(r),
            )
            # Identity sentinel: keeps forward in the upsample_layer branch,
            # avoiding F.interpolate which does not support "pixel_shuffle" as mode.
            self.upsample_layer: nn.Module | None = nn.Identity()
        else:
            self.cls_head = nn.Conv2d(self.channels, num_classes, kernel_size=1)
            if upsample_mode == "transposed_conv":
                if not patch_size:
                    raise ValueError("transposed_conv upsample_mode requires patch_size.")
                self.upsample_layer = _build_transposed_upsample(num_classes, int(patch_size))
            else:
                self.upsample_layer = None

    def _upsample(self, x: Tensor, output_size) -> Tensor:
        output_size = tuple(int(v) for v in output_size)
        if self.upsample_layer is not None:
            x = self.upsample_layer(x)
            if tuple(x.shape[-2:]) != output_size:
                x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        else:
            align_corners = False if self.upsample_mode in _ALIGN_CORNERS_MODES else None
            x = F.interpolate(x, size=output_size, mode=self.upsample_mode,
                               align_corners=align_corners)
        return x

    def forward(
        self,
        features: Sequence[Tensor],
        output_size: tuple[int, int] | torch.Size | None = None,
    ) -> Tensor:
        if len(features) != len(self.projections):
            raise ValueError(
                f"Expected {len(self.projections)} feature taps, got {len(features)}."
            )

        projected = [
            F.relu(proj(f), inplace=True)
            for proj, f in zip(self.projections, features)
        ]
        x = torch.cat(projected, dim=1)
        x = self.fusion(x)
        x = self.cls_head(x)

        if output_size is None:
            raise ValueError(
                "MLPFusionHead.forward requires output_size (his-bench segmentors "
                "always pass it; pangaea's original fell back to the input image "
                "shape, which isn't available here since the decoder no longer "
                "owns the encoder call)."
            )
        return self._upsample(x, output_size)
