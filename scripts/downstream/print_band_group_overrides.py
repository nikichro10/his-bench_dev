"""Print Hydra CLI overrides for each (sensor, band-group) combination used by
run_wavelength_ablation.sh.

Deliberately avoids importing anything under `src.` (e.g.
src.utils.spectral_metadata.resolve_band_group, which has the same
wavelength-range logic as this script). `src.utils`'s package __init__
eagerly imports torch/lightning, and on this cluster that chain can block
for a long time (observed: a plain-python process stuck on
`cxiBlockingMutexAcquire`, an HPE Slingshot/CXI network-driver mutex --
apparently only triggered outside of an actual SLURM job allocation, since
real training jobs import the same chain successfully). This script only
needs two small YAML sensor configs and their wavelength CSVs, so it reads
them directly with plain `yaml`/`csv` instead, keeping submission fast and
independent of that hang.

BAND_GROUP_RANGES_NM below must stay identical to
src.utils.spectral_metadata.BAND_GROUP_RANGES_NM -- duplicated here only to
avoid the import above, not because the boundaries differ.

Also emits model.probe_in_chans=<n> -- configs/model/{segmentation,multilabel_segmentation}.yaml
hardcode probe_in_chans: ${data.sensor_config.num_bands} (the FULL sensor band
count) for the eager multi-tap decoder shape probe (see
src.models.model_utils.infer_multitap_decoder_info /
_infer_backbone_probe_shape). That interpolation doesn't know about a band
subset, so without this override the probe builds a dummy input at the full
band count while the backbone's dynamic patch embed is correctly sized for
the subset -- a Conv2d in-channels mismatch caught by an end-to-end test
before mass submission, not a hypothetical.

Two wave_list variants are emitted (nm and um) because backbones disagree on
units: dofa_b/dofav2_b/carl auto-derive their wave_list as
band_centers_nm * 0.001 (micrometers) when no override is given, while
copernicus_fm_b/panopticon_b use band_centers_nm directly (nanometers, no
conversion) -- see each backbone's __init__. An explicit wave_list override
is used AS-IS with no conversion, so passing nm to a um-expecting backbone
silently feeds wavelength values ~1000x too large into its wavelength
positional encoding. Caught via an end-to-end test: dofa_b/rgb trained
without error but train_loss was NaN from literally the first logged step,
traced to exactly this unit mismatch.

Also emits a bandwidth_list override (nm, no unit variant needed --
copernicus_fm_b is the only consumer and it takes band_widths_nm directly).
configs/backbone/copernicus_fm_b.yaml declares bandwidth_list separately from
wave_list (both default null -> auto-derive from the FULL SensorRegistry
metadata when not given). Overriding wave_list alone without also
overriding bandwidth_list would leave self.bw_list at the full sensor's 244
widths while self.wv_list is correctly subsetted -- a length mismatch inside
copernicus_fm's Fourier wavelength/bandwidth embedding. Only apply this
override for copernicus_fm_b; other backbones don't declare a
bandwidth_list field, so passing it would hit the same "not in struct"
error +data.band_indices needed a + for.

Output: one TAB-separated line per (sensor, group) combo:
    <sensor>\\t<group>\\t<common overrides>\\t<wave_list override, nm>\\t<wave_list override, um>\\t<bandwidth_list override, nm>

Each wave_list/bandwidth_list override is wrapped in real (unescaped) double
quotes so that, after being captured into a bash variable and later
interpolated into a command string that gets `eval`-reparsed, the embedded
`{...}` survives bash's brace expansion (which would otherwise split on the
commas inside the list) -- same mechanism already used for
SLURM_EXCLUDE='\"...\"' elsewhere in these scripts, adapted for a value that
arrives via command substitution rather than being typed as literal bash
source.
"""

import csv
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

SENSOR_YAML_PATHS = {
    "emit": REPO_ROOT / "configs/sensor/emit.yaml",
    "enmap": REPO_ROOT / "configs/sensor/enmap.yaml",
}

BAND_GROUP_RANGES_NM = {
    "vis": (380.0, 700.0),
    "vnir": (380.0, 1000.0),
    "swir": (1000.0, 2600.0),
}

GROUPS = ["rgb", "vis", "vnir", "swir"]


def load_sensor_wavelengths(sensor: str) -> tuple[list[int], list[float], list[float]]:
    """Return (rgb_indices, band_centers_nm, band_widths_nm) for the processed band view."""
    yaml_path = SENSOR_YAML_PATHS[sensor]
    with yaml_path.open() as f:
        cfg = yaml.safe_load(f)

    rgb_indices = [int(i) for i in cfg["rgb_indices"]]
    metadata_path = REPO_ROOT / cfg["spectral"]["processed_band_metadata_path"]
    with metadata_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    wavelengths = [float(row["wavelength_nm"]) for row in rows]
    widths = [float(row["width_nm"]) for row in rows]

    num_bands = int(cfg["num_bands"])
    if len(wavelengths) != num_bands:
        raise ValueError(
            f"{sensor}: processed metadata has {len(wavelengths)} rows, "
            f"expected num_bands={num_bands}."
        )
    return rgb_indices, wavelengths, widths


def resolve_group(
    group: str, rgb_indices: list[int], wavelengths: list[float], widths: list[float]
) -> tuple[list[int], list[float], list[float]]:
    if group == "rgb":
        indices = rgb_indices
    elif group in BAND_GROUP_RANGES_NM:
        lo, hi = BAND_GROUP_RANGES_NM[group]
        indices = [i for i, w in enumerate(wavelengths) if lo <= w < hi]
    else:
        raise ValueError(f"Unknown band group '{group}'.")
    return indices, [wavelengths[i] for i in indices], [widths[i] for i in indices]


def _dict_list_override(field: str, sensor: str, values: list[float]) -> str:
    vals_str = "[" + ",".join(f"{v:.6f}" for v in values) + "]"
    return f'model.backbone_config.{field}="{{{sensor}:{vals_str}}}"'


def wave_list_override(sensor: str, wavelengths_nm: list[float], unit: str) -> str:
    if unit == "nm":
        values = wavelengths_nm
    elif unit == "um":
        values = [w * 0.001 for w in wavelengths_nm]
    else:
        raise ValueError(f"Unknown unit '{unit}'.")
    return _dict_list_override("wave_list", sensor, values)


for sensor in SENSOR_YAML_PATHS:
    rgb_indices, wavelengths, widths = load_sensor_wavelengths(sensor)
    for group in GROUPS:
        indices, group_wavelengths, group_widths = resolve_group(
            group, rgb_indices, wavelengths, widths
        )
        idx_str = "[" + ",".join(str(i) for i in indices) + "]"
        common = f"+data.band_indices={idx_str} model.probe_in_chans={len(indices)}"
        wl_nm = wave_list_override(sensor, group_wavelengths, "nm")
        wl_um = wave_list_override(sensor, group_wavelengths, "um")
        bw_nm = _dict_list_override("bandwidth_list", sensor, group_widths)
        print(f"{sensor}\t{group}\t{common}\t{wl_nm}\t{wl_um}\t{bw_nm}")
