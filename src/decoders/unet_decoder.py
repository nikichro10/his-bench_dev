# Adapted from https://github.com/VMarsocci/pangaea-bench/blob/main/pangaea/decoders/unet.py
# Standard 4-level UNet decoder (transpose-conv upsampling + skip
# connections), paired with src.backbones.unet_backbone.UNetBackbone for a
# fully-supervised baseline.

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class UNetDecoder(nn.Module):
    """
    UNet decoder for a fully-supervised semantic segmentation baseline.

    Args:
        topology (Sequence[int]): feature channels at each stage, must match
            the paired ``UNetBackbone``'s topology.
        num_classes (int): number of output classes.
        num_input_features (int, optional): the generic decoder-instantiation
            path in ``semantic_segmentation_module.py`` injects the
            backbone's ``num_features`` under this name for any decoder it
            doesn't special-case (upernet/multiscale_conv/multitap/mlp_fusion).
            UNetDecoder already gets its channel counts from ``topology``, so
            this is accepted only for interface compatibility and, when
            given, checked against ``topology[-1]`` (the backbone's deepest
            pyramid level) to catch a topology mismatch early.
    """

    def __init__(
        self,
        topology: Sequence[int] = (64, 128, 256, 512),
        num_classes: int = 1,
        num_input_features: int | None = None,
    ):
        super().__init__()
        self.topology = list(topology)
        self.num_classes = num_classes
        self.in_channels = self.topology[0]

        if num_input_features is not None and num_input_features != self.topology[-1]:
            raise ValueError(
                f"UNetDecoder topology[-1]={self.topology[-1]} does not match "
                f"the backbone's reported num_features={num_input_features} -- "
                "UNetBackbone and UNetDecoder must be configured with the same topology."
            )

        self.decoder = UNet_Decoder(self.topology)
        self.conv_seg = OutConv(self.topology[0], self.num_classes)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        feat = self.decoder(list(features))
        return self.conv_seg(feat)


class UNet_Decoder(nn.Module):
    def __init__(self, topology: Sequence[int]):
        super().__init__()
        self.topology = topology

        n_layers = len(topology)
        up_topo = [topology[0]]
        up_dict = OrderedDict()

        for idx in range(n_layers):
            is_not_last_layer = idx != n_layers - 1
            out_dim = topology[idx + 1] if is_not_last_layer else topology[idx]
            up_topo.append(out_dim)

        for idx in reversed(range(n_layers)):
            is_not_last_layer = idx != 0
            x1_idx = idx
            x2_idx = idx - 1 if is_not_last_layer else idx
            in_dim = up_topo[x1_idx] * 2
            out_dim = up_topo[x2_idx]
            up_dict[f"up{idx + 1}"] = Up(in_dim, out_dim, DoubleConv)

        self.up_seq = nn.ModuleDict(up_dict)

    def forward(self, features: list) -> torch.Tensor:
        x1 = features.pop(0)
        for idx, layer in enumerate(self.up_seq.values()):
            x2 = features[idx]
            x1 = layer(x1, x2)
        return x1


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


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, conv_block):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        self.conv = conv_block(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.detach().size()[2] - x1.detach().size()[2]
        diffX = x2.detach().size()[3] - x1.detach().size()[3]
        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.conv(x)
