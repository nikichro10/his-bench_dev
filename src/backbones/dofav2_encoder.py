# Adapted from: https://github.com/VMarsocci/pangaea-bench (dofav2_encoder.py)
# Which is in turn based on: https://github.com/zhu-xlab/DOFA
# Weights downloaded from: https://huggingface.co/XShadow/DOFA/resolve/main/dofav2_vit_base_e150.pth

import math
from functools import partial
from logging import Logger
from pathlib import Path
from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer

from ..backbones.base import Encoder
from src.backbones.dofa_encoder import Dynamic_MLP_OFA
from src.utils.sensor_registry import SensorRegistry
from src.utils.spectral_metadata import load_spectral_metadata


class DOFAv2_Encoder(Encoder):
    """
    DOFAv2 encoder matching the official implementation structure:
      - Uses timm's VisionTransformer with dynamic_img_size=True for native
        positional-embedding interpolation at any input resolution, instead of
        DOFA v1's fixed, non-interpolated sin-cos pos_embed.
      - LayerScale enabled via init_values=1e-5 (not present in v1).
      - patch_embed.proj deleted; Dynamic_MLP_OFA (shared with DOFA v1) is used
        for wavelength-conditioned patch embedding instead.
      - Intermediate features collected pre-norm, matching the official forward.

    Paper: https://arxiv.org/pdf/2403.15356 (DOFA); v2 checkpoint per
    https://huggingface.co/XShadow/DOFA.
    """

    def __init__(
        self,
        encoder_weights: Union[str, List[Path]],
        input_bands: Optional[dict[str, list[str]]],
        input_size: int,
        embed_dim: int,
        output_dim: Union[int, List[int]],
        output_layers: Union[int, List[int]],
        wave_list: Optional[dict[str, list[float]]],
        patch_size: int = 14,
        depth: int = 12,
        num_heads: int = 12,
        wv_planes: int = 128,
        mlp_ratio: float = 4.0,
        use_norm: bool = False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        sensor: str = "enmap",
        sensor_config_name: Optional[str] = None,
        reinit_weight_generator: bool = False,
    ):
        super().__init__(
            model_name="dofav2_encoder",
            encoder_weights=encoder_weights,
            input_bands=input_bands,
            input_size=input_size,
            embed_dim=embed_dim,
            output_layers=output_layers,
            output_dim=output_dim,
            multi_temporal=False,
            multi_temporal_output=False,
            pyramid_output=False,
        )

        self.output_layers = output_layers
        self.img_size = input_size
        self.patch_size = patch_size
        self.wave_list = wave_list
        self.reinit_weight_generator = reinit_weight_generator
        self.use_norm = use_norm
        self.sensor = str(sensor_config_name or sensor).lower()
        if self.wave_list is not None:
            self.wv_list = self.wave_list[self.sensor]
        else:
            sensor_cfg = SensorRegistry.get(self.sensor)
            metadata = load_spectral_metadata(sensor_cfg, view="processed")
            self.wv_list = (metadata.band_centers_nm * 0.001).astype("float32").tolist()
        self.num_features = embed_dim
        self.in_chans = len(self.wv_list)

        # Dynamic_MLP_OFA replaces the standard conv patch embedding.
        self.patch_embed = Dynamic_MLP_OFA(
            wv_planes=wv_planes, inter_dim=128, kernel_size=patch_size, embed_dim=embed_dim
        )

        # VisionTransformer provides cls_token, pos_embed, blocks, and norm.
        # dynamic_img_size=True makes _pos_embed accept (B, H, W, C) and
        # interpolate pos_embed automatically for non-native input sizes.
        # init_values=1e-5 enables LayerScale, required by DOFAv2 checkpoints.
        self.model = VisionTransformer(
            img_size=input_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            init_values=1e-5,
            num_classes=0,
            dynamic_img_size=True,
            norm_layer=norm_layer,
        )
        # Remove the conv proj; we use Dynamic_MLP_OFA instead. The PatchEmbed
        # object itself is kept so _pos_embed can still read grid_size.
        del self.model.patch_embed.proj

    def _embed_and_prepare(self, x: torch.Tensor):
        """Wavelength-conditioned patch embedding + timm's dynamic pos_embed path.

        Shared by forward() and get_intermediate_layers() so both stay in sync.
        """
        self.waves = torch.tensor(self.wv_list, device=x.device).float()
        x, _ = self.patch_embed(x, self.waves)

        # Reshape to (B, H, W, C) so timm's _pos_embed can infer grid size and
        # interpolate pos_embed when the input differs from input_size.
        B, HW, C = x.shape
        hw = int(math.sqrt(HW))
        x = x.view(B, hw, hw, C)

        x = self.model._pos_embed(x)
        x = self.model.patch_drop(x)
        x = self.model.norm_pre(x)
        return x, hw

    def forward(self, x: torch.Tensor):
        x, hw = self._embed_and_prepare(x)

        # Collect intermediate features pre-norm, matching the official forward.
        output = []
        for i, blk in enumerate(self.model.blocks):
            x = blk(x)
            if i in self.output_layers:
                out = (
                    x[:, 1:]
                    .permute(0, 2, 1)
                    .view(x.shape[0], -1, hw, hw)
                    .contiguous()
                )
                output.append(out)

        self.model.norm(x)

        if len(self.output_layers) == 0:
            if hasattr(self, "classifier"):
                return self.classifier(x[:, 0])
            return x[:, 0]
        return output

    def _resolve_output_layers(
        self,
        n: Optional[Union[int, Sequence[int]]] = None,
    ) -> list[int]:
        num_blocks = len(self.model.blocks)
        if n is None:
            layers = self.output_layers
        elif isinstance(n, int):
            layers = list(range(num_blocks - n, num_blocks)) if n > 0 else self.output_layers
        else:
            layers = list(n)
        if isinstance(layers, int):
            return [layers]
        return [int(layer) for layer in layers]

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Optional[Union[int, Sequence[int]]] = None,
        reshape: bool = False,
        return_prefix_tokens: bool = False,
        norm: bool = False,
    ):
        x, hw = self._embed_and_prepare(x)

        take_layers = set(self._resolve_output_layers(n))
        num_blocks = len(self.model.blocks)

        output = []
        prefix_output = []
        for i, blk in enumerate(self.model.blocks):
            x = blk(x)
            if i == num_blocks - 1 and norm:
                x = self.model.norm(x)
            if i in take_layers:
                out = x[:, 1:]
                prefix_output.append(x[:, :1])
                if reshape:
                    out = (
                        out.permute(0, 2, 1)
                        .view(out.shape[0], -1, hw, hw)
                        .contiguous()
                    )
                output.append(out)

        if return_prefix_tokens:
            return tuple(zip(output, prefix_output))
        return output

    def set_wv_list_from_data_bands(self, data_bands: dict[str, list[str]]) -> None:
        """Restrict wv_list to bands actually present in the dataset.
        Call this when BandPadding is NOT used so DOFAv2 receives only real bands
        with their true wavelengths rather than zero-padded phantom channels.
        """
        wv_list = [
            self.wave_list[m][bi]
            for m, b in self.input_bands.items()
            for bi in b
            if m in data_bands and bi in data_bands[m]
        ]
        if not wv_list:
            raise ValueError("No wavelengths remain after filtering to available data bands.")
        self.wv_list = wv_list

    def load_state_dict(self, state_dict, *args, **kwargs):
        """Overridden so the reinit-hypernetwork ablation reinitializes
        patch_embed's weight-generator AFTER any pretrained weights land --
        see DOFA_Encoder.load_state_dict's docstring (Dynamic_MLP_OFA /
        reinit_hypernetwork() are shared with DOFA v1, imported directly)."""
        result = super().load_state_dict(state_dict, *args, **kwargs)
        if self.reinit_weight_generator:
            self.patch_embed.reinit_hypernetwork()
        return result

    def load_encoder_weights(self, logger: Logger) -> None:
        pretrained_model = torch.load(self.encoder_weights, map_location="cpu", weights_only=False)
        # No prefix stripping needed: self.model = VisionTransformer, so
        # named_parameters() naturally yields 'model.blocks.*', 'model.pos_embed',
        # etc., which directly match the DOFAv2 checkpoint's keys.
        k = pretrained_model.keys()
        pretrained_encoder = {}
        incompatible_shape = {}
        missing = {}
        for name, param in self.named_parameters():
            if name not in k:
                missing[name] = param.shape
            elif pretrained_model[name].shape != param.shape:
                incompatible_shape[name] = (param.shape, pretrained_model[name].shape)
            else:
                pretrained_encoder[name] = pretrained_model[name]

        self.load_state_dict(pretrained_encoder, strict=False)
        self.parameters_warning(missing, incompatible_shape, logger)


def DofaV2BasePatch14(**kwargs):
    kwargs.pop("patch_size", None)
    model = DOFAv2_Encoder(
        patch_size=14, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), output_dim=768, **kwargs)
    return model


def DofaV2LargePatch14(**kwargs):
    kwargs.pop("patch_size", None)
    model = DOFAv2_Encoder(
        patch_size=14, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), output_dim=1024, **kwargs)
    return model
