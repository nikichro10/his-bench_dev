"""UniverSat (AnySat v2) wrapper for comparison experiments.

https://github.com/gastruc/UniverSat

Architecturally different from every other backbone in this repo: UniverSat's
``encode()`` returns a single dense feature grid at a caller-chosen output
resolution, not per-transformer-block taps. There is no way to pull
intermediate features the way UPerNet/MLPFusion need (both require 4 tap
layers), so this backbone is only wired up for the single-tap
``model_type="vit_seg"``/``"conv_seg"`` path (``ConvSegmentor`` +
``conv_head``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn

from src.utils.sensor_registry import SensorRegistry
from src.utils.spectral_metadata import load_spectral_metadata


class UniverSatBackbone(nn.Module):
    """Thin adapter around the UniverSat torch-hub model.

    UniverSat's own ``modality_registry.py`` registers "enmap" (202 bands)
    with wavelengths/resolution that match his-bench's own EnMAP metadata
    almost exactly, but does not register "desis" at all, and registers
    "EO1" with a different band count (175) than his-bench's own EO1
    processing (163). Rather than rely on UniverSat's built-in registry for
    any sensor, wavelengths/input_res/subpatches are always supplied
    explicitly here from his-bench's own SensorRegistry.

    This is architecturally safe: the modality *name* string passed to
    ``encode()`` is never used to index a learned per-sensor embedding
    inside ``UniversalPatchEncoder.forward()`` -- only the wavelength values
    supplied alongside it are (numeric wavelengths go through Learnable
    Fourier Features; only a small closed set of SAR/DEM channel *codes*
    ('VV'/'VH'/'HH'/'HV'/'Ratio_VV_VH'/'Ratio_HH_HV'/'nDEM'/'DSM') route to
    fixed learned parameters, and hyperspectral sensors never use those).
    So a modality name unseen at pretraining (e.g. "desis") is safe as long
    as real wavelengths are supplied, which they always are here.
    """

    def __init__(
        self,
        sensor: str = "enmap",
        sensor_config_name: Optional[str] = None,
        hub_repo: str = "gastruc/UniverSat",
        hf_repo_id: str = "g-astruc/UniverSat",
        img_size: int = 224,
        patch_size: int = 1,
        latent_patch_size: int = 2,
        num_features: int = 768,
        compile: Union[bool, str] = False,
        pretrained: bool = True,
        modality_name: Optional[str] = None,
        subpatches: int = 1,
        trust_repo: Optional[bool] = None,
    ) -> None:
        super().__init__()
        self.sensor = str(sensor_config_name or sensor).lower()
        self.img_size = int(img_size)
        # patch_size: pixels per OUTPUT token (what ConvSegmentor reshapes
        # into) -- decoupled from the model's own input tokenization scale.
        # Faithful to UniverSat's own EnMAP eval config (EnmapCdl.yaml:
        # out_patch_size=1, num_patches=16384=128^2), which requests a full
        # dense per-pixel output grid via the model's "Bilinear_out"/"CA_Sub"
        # sub-patch-recovery mechanism -- patch_size=1 means "one output
        # token per pixel", not "no output resampling" (that was the old,
        # unfaithful default of matching the latent grid 1:1).
        self.patch_size = int(patch_size)
        # latent_patch_size: pixels per INPUT/latent token, i.e. what
        # controls "scale" (patch_size in metres = latent_patch_size *
        # sensor resolution) fed to UniverSat's own encode(). Faithful
        # default of 2 matches EnmapCdl.yaml's scale_base=6 at EnMAP's 30m
        # resolution (60m / 30m = 2 px/token -> 64x64 latent grid at
        # img_size=128), decoupled from the output grid.
        self.latent_patch_size = int(latent_patch_size)
        self.num_features = int(num_features)

        sensor_cfg = SensorRegistry.get(self.sensor)
        metadata = load_spectral_metadata(sensor_cfg, view="processed")
        self.in_chans = metadata.num_bands
        # UniverSat's spectral Fourier embedding expects micrometers.
        self.wavelengths_um = [c / 1000.0 for c in metadata.band_centers_nm]
        self.input_res_m = float(sensor_cfg.get("resolution", 30.0))
        self.subpatches = int(subpatches)
        # Dict-key label only (see class docstring) -- reusing the sensor
        # name keeps debug output readable; it need not be a name UniverSat's
        # own registry recognises.
        self.modality_name = str(modality_name or self.sensor)

        hub_repo_path = Path(hub_repo).expanduser()
        load_kwargs: dict = {"compile": compile}
        if trust_repo is not None:
            load_kwargs["trust_repo"] = trust_repo

        if hub_repo_path.exists():
            source = "local"
            repo_arg = str(hub_repo_path)
        else:
            source = "github"
            repo_arg = hub_repo

        if pretrained:
            self.model = torch.hub.load(
                repo_arg, "from_pretrained", repo_id=hf_repo_id, source=source, **load_kwargs
            )
        else:
            self.model = torch.hub.load(
                repo_arg, "universat", pretrained=False, source=source, **load_kwargs
            )
        self.model.eval()

    def _encode(self, x: torch.Tensor, output_side: int) -> torch.Tensor:
        data = {self.modality_name: x}
        patch_size_m = self.latent_patch_size * self.input_res_m
        features, _ = self.model.encode(
            data,
            patch_size=patch_size_m,
            output_grid=output_side,
            wavelengths={self.modality_name: self.wavelengths_um},
            input_res={self.modality_name: self.input_res_m},
            subpatches={self.modality_name: self.subpatches},
        )
        return features  # (B, output_side**2, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        side = int(x.shape[-2]) // self.patch_size
        return self._encode(x, side)

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence[int]] = 1,
        reshape: bool = False,
        return_prefix_tokens: bool = False,
        norm: bool = False,
    ):
        """Single-tap only: always returns a 1-tuple. ``n``/``norm`` are
        accepted for interface compatibility with the other backbones but
        have no effect -- UniverSat exposes no intermediate transformer
        blocks and its own final norm is already baked into ``encode()``."""
        side = int(x.shape[-2]) // self.patch_size
        tokens = self._encode(x, side)  # (B, side*side, embed_dim)

        if reshape:
            b, _, d = tokens.shape
            tokens = tokens.permute(0, 2, 1).reshape(b, d, side, side).contiguous()

        result = (tokens, None) if return_prefix_tokens else tokens
        return (result,)
