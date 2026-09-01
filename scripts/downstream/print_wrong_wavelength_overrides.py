"""Print Hydra CLI overrides for the wrong-wavelength ablation used by
run_wrong_wavelength_ablation.sh / run_wrong_wavelength_universat.sh.

Different question than the band-restriction ablations
(print_band_group_overrides.py / print_random_band_overrides.py): those keep
wave_list correct and vary WHICH/HOW MANY bands the model sees. This one
keeps every band (full spectrum, same as the existing full-band baseline --
no data.band_indices/probe_in_chans override at all, since band count never
changes here) and instead feeds each wavelength-conditioned backbone
DELIBERATELY WRONG center-wavelength metadata for the same real pixel data.
If a backbone's wavelength conditioning is actually load-bearing (not just a
fixed per-channel embedding it could equally have learned positionally),
performance should drop when the labels it's given for "which wavelength is
this channel" are wrong -- directly analogous to the user's earlier
multispectral pangaea-bench experiment feeding wrong central wavelengths.

Two conditions, per sensor:

  wrong_permuted_s{0,1,2}: the TRUE set of per-band wavelength values, but
    randomly reassigned to different bands (random.Random(seed).sample, a
    full derangement-agnostic permutation of range(num_bands)) -- band i's
    pixel data is real, but the backbone is told it's band perm[i]'s
    wavelength. 3 seeds (same convention as
    print_random_band_overrides.py's draws) so a headline effect can be
    checked for robustness across which specific permutation landed, rather
    than resting on one lucky/unlucky shuffle.

  wrong_constant: every band is told the SAME wavelength -- the sensor's
    median true wavelength (a physically-plausible "middle of the spectrum"
    value, not an edge case like the min or max). Removes all cross-channel
    differentiation from the conditioning signal entirely: if performance
    barely drops, the backbone isn't using wavelength identity to
    distinguish channels at all. No seed -- deterministic per sensor.

Band WIDTHS are the TRUE, unpermuted per-band widths (in original band
order) -- always emitted, so this stays a single-factor test of
center-wavelength correctness. This does NOT auto-derive on its own for
copernicus_fm_b: CopernicusFMBasePatch16.__init__
(src/backbones/copernicus_fm_encoder.py) only auto-derives wv_list/bw_list
from SensorRegistry TOGETHER, when wave_list is None. The instant wave_list
is overridden (as it always is here), it unconditionally does
self.bw_list = self.bandwidth_list[self.sensor] -- if bandwidth_list wasn't
ALSO explicitly passed, that's `None[self.sensor]`, a TypeError. Caught via
an end-to-end test before mass submission (this file originally omitted
bandwidth_list entirely, assuming independent auto-derive -- confirmed wrong
by that crash). run_wrong_wavelength_ablation.sh passes this override only
for copernicus_fm_b, matching the exact convention already established in
print_band_group_overrides.py/print_random_band_overrides.py and their
consumer scripts.

Deliberately avoids importing src.* -- same reason as
print_band_group_overrides.py (torch/lightning import chain can hang on the
login node outside a SLURM allocation).

Output: one TAB-separated line per (sensor, condition) combo:
    <sensor>\\t<condition>\\t<wave_list override, nm>\\t<wave_list override, um>\\t<bandwidth_list override, nm>

Quoting rationale for wave_list/bandwidth_list is identical to
print_band_group_overrides.py -- see that script's header.
"""

import csv
import random
import statistics
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

SENSOR_YAML_PATHS = {
    "emit": REPO_ROOT / "configs/sensor/emit.yaml",
    "enmap": REPO_ROOT / "configs/sensor/enmap.yaml",
}

SEEDS = [0, 1, 2]


def load_sensor_wavelengths(sensor: str) -> tuple[list[float], list[float]]:
    """Return (band_centers_nm, band_widths_nm) for the processed band view."""
    yaml_path = SENSOR_YAML_PATHS[sensor]
    with yaml_path.open() as f:
        cfg = yaml.safe_load(f)

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
    return wavelengths, widths


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
    wavelengths, widths = load_sensor_wavelengths(sensor)
    num_bands = len(wavelengths)
    # Always the TRUE, unpermuted widths -- band widths are never corrupted
    # by either condition below, only center wavelengths are.
    bw_nm = _dict_list_override("bandwidth_list", sensor, widths)

    for seed in SEEDS:
        rng = random.Random(seed)
        perm = rng.sample(range(num_bands), num_bands)
        wrong_wavelengths = [wavelengths[i] for i in perm]
        condition_name = f"wrong_permuted_s{seed}"
        wl_nm = wave_list_override(sensor, wrong_wavelengths, "nm")
        wl_um = wave_list_override(sensor, wrong_wavelengths, "um")
        print(f"{sensor}\t{condition_name}\t{wl_nm}\t{wl_um}\t{bw_nm}")

    median_wavelength = statistics.median(wavelengths)
    constant_wavelengths = [median_wavelength] * num_bands
    wl_nm = wave_list_override(sensor, constant_wavelengths, "nm")
    wl_um = wave_list_override(sensor, constant_wavelengths, "um")
    print(f"{sensor}\twrong_constant\t{wl_nm}\t{wl_um}\t{bw_nm}")
