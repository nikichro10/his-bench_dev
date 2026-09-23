# Adapted from https://github.com/VMarsocci/pangaea-bench/blob/main/pangaea/encoders/unet_encoder.py
# Standard 4-level UNet encoder, trained from scratch (no pretrained weights),
# used as a fully-supervised comparison baseline alongside the frozen
# foundation-model backbones.

from collections import OrderedDict
from typing import List, Sequence

import torch
import torch.nn as nn


class UNetBackbone(nn.Module):
    """
    UNet encoder for a fully-supervised baseline. Single time-frame,
    single-sensor input (the hyperspectral band stack for whichever sensor
    the task uses), no pretrained weights.

    Args:
        in_chans (int): number of input channels (bands), e.g. wired from
            ``${sensor.num_bands}``.
        topology (Sequence[int]): number of feature channels at each stage
            of the encoder.
        img_size (int): training image size (unused internally, kept for
            config-interface consistency with the other backbones).
        name (str): backbone name, for logging/config bookkeeping only.
    """

    def __init__(
        self,
        in_chans: int,
        topology: Sequence[int] = (64, 128, 256, 512),
        img_size: int = 224,
        name: str = "unet_b",
    ):
        super().__init__()
        self.in_chans = in_chans
        self.topology = list(topology)
        self.img_size = img_size
        self.name = name
        self.num_features = self.topology[-1]

        self.in_conv = InConv(self.in_chans, self.topology[0], DoubleConv)
        self.encoder = UNet_Encoder(self.topology)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Returns the multi-scale feature pyramid, deepest first (matches
        the skip-connection order ``UNetDecoder`` expects)."""
        feat = self.in_conv(x)
        return self.encoder(feat)


class UNet_Encoder(nn.Module):
    """Downsampling path: one `Down` (maxpool + DoubleConv) block per
    topology stage."""

    def __init__(self, topology: Sequence[int]):
        super().__init__()
        self.topology = topology

        down_topo = topology
        down_dict = OrderedDict()
        n_layers = len(down_topo)

        for idx in range(n_layers):
            is_not_last_layer = idx != n_layers - 1
            in_dim = down_topo[idx]
            out_dim = down_topo[idx + 1] if is_not_last_layer else down_topo[idx]
            down_dict[f"down{idx + 1}"] = Down(in_dim, out_dim, DoubleConv)
        self.down_seq = nn.ModuleDict(down_dict)

    def forward(self, x1: torch.Tensor) -> List[torch.Tensor]:
        inputs = [x1]
        for layer in self.down_seq.values():
            inputs.append(layer(inputs[-1]))

        inputs.reverse()
        return inputs


class DoubleConv(nn.Module):
    """(conv => BN => ReLU) * 2"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch, conv_block):
        super().__init__()
        self.conv = conv_block(in_ch, out_ch)

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, conv_block):
        super().__init__()
        self.mpconv = nn.Sequential(nn.MaxPool2d(2), conv_block(in_ch, out_ch))

    def forward(self, x):
        return self.mpconv(x)
