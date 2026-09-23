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

Three conditions, per sensor:

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

  wrong_jitter_s{0} / wrong_jitter_50_100_s{0}: a much milder corruption
    than the two above -- each band keeps its OWN true wavelength as an
    anchor, just nudged by a small, independently-drawn perturbation
    (magnitude uniform in [20, 50]nm for wrong_jitter, [50, 100]nm for
    wrong_jitter_50_100; random sign per band -- so no band ever gets a
    near-zero nudge, and perturbations aren't all the same direction the way
    a single global calibration shift would be). Tests a different, more
    realistic failure mode than wrong_permuted/wrong_constant: not "the
    backbone was told a completely wrong band identity" but "the backbone's
    wavelength metadata has ordinary calibration-drift-sized noise on it" --
    does it need precisely correct wavelengths, or just approximately
    correct ones? wrong_jitter_50_100 pushes that same question with a
    larger, still-bounded magnitude to see whether the near-total tolerance
    of [20,50]nm noise degrades once the offset roughly doubles. Only 1 seed
    each for now (JITTER_RANGES below defines each range's own seed list),
    unlike wrong_permuted's 3 -- extend a range's seed list the same way if
    more draws are wanted later.

  shuffled_s{0}: a fundamentally different question from the three above --
    those all keep the pixel DATA in its original band order and corrupt
    only the wavelength LABEL. This one shuffles the data AND the labels by
    the exact SAME permutation (SHUFFLE_SEEDS below -- reuses
    wrong_permuted_s0's own permutation for direct comparability, not a
    fresh draw), so the (data, wavelength) pairing at every position stays
    CORRECT, just not in the sensor's natural ascending order anymore.
    wrong_permuted asks "does the backbone use wavelength info at all";
    shuffled asks "does the backbone need canonical band ORDER even when its
    wavelength conditioning is telling the truth at every position" -- if
    wavelength conditioning is truly channel-order-agnostic, this should
    barely hurt; a real drop would reveal a hidden order-dependence (e.g. a
    conv-like structure implicitly assuming spectrally-adjacent neighbors)
    underneath the conditioning. Only 1 seed for now (SHUFFLE_SEEDS below).
    Needs a `+data.band_indices=<permutation>` override too (unlike the
    other three conditions) -- reuses the exact same mechanism
    print_random_band_overrides.py already uses for subset selection, just
    with a full, UNSORTED permutation of all num_bands indices instead of a
    sorted subset, so BaseSegmentationDataset's
    `tensor_image[self.band_indices, :, :]` reorders channels instead of
    subsetting them. Band count never changes, so (unlike the subset
    scripts) no model.probe_in_chans override is needed -- the decoder's
    default (full sensor num_bands) is already numerically correct.

Band WIDTHS are the TRUE, unpermuted per-band widths (in original band
order) for wrong_permuted/wrong_constant/wrong_jitter -- always emitted, so
those three stay a single-factor test of center-wavelength correctness. For
shuffled, widths are reordered by the SAME permutation as the data/
wavelengths (so the (data, wavelength, bandwidth) triple stays fully
consistent at every position, matching the "correct data, wrong order"
design) -- a deliberate difference from the other three, not an oversight.
Bandwidth_list does NOT auto-derive on its own for copernicus_fm_b:
CopernicusFMBasePatch16.__init__ (src/backbones/copernicus_fm_encoder.py)
only auto-derives wv_list/bw_list TOGETHER from SensorRegistry, when
wave_list is None. The instant wave_list is overridden (as it always is
here), it unconditionally does self.bw_list = self.bandwidth_list[self.sensor]
-- if bandwidth_list wasn't ALSO explicitly passed, that's `None[self.sensor]`,
a TypeError. Caught via an end-to-end test before mass submission (this file
originally omitted bandwidth_list entirely, assuming independent auto-derive
-- confirmed wrong by that crash). run_wrong_wavelength_ablation.sh passes
this override only for copernicus_fm_b, matching the exact convention
already established in print_band_group_overrides.py/
print_random_band_overrides.py and their consumer scripts.

Deliberately avoids importing src.* -- same reason as
print_band_group_overrides.py (torch/lightning import chain can hang on the
login node outside a SLURM allocation).

Output: one TAB-separated line per (sensor, condition) combo:
    <sensor>\\t<condition>\\t<wave_list override, nm>\\t<wave_list override, um>\\t<bandwidth_list override, nm>\\t<band_indices override, or empty>

The band_indices field is empty for wrong_permuted/wrong_constant/
wrong_jitter (band count/order never changes for those) and populated only
for shuffled_s0.

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

# wrong_jitter variants -- (condition label, min magnitude nm, max magnitude
# nm, seeds). Only 1 seed each for now, see module docstring.
JITTER_RANGES = [
    ("wrong_jitter", 20.0, 50.0, [0]),
    ("wrong_jitter_50_100", 50.0, 100.0, [0]),
]

# shuffled -- only 1 seed for now. MUST be a subset of SEEDS: reuses
# wrong_permuted_s{seed}'s own permutation (same random.Random(seed).sample
# call) for direct comparability, not a fresh draw.
SHUFFLE_SEEDS = [0]


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


def band_indices_override(indices: list[int]) -> str:
    idx_str = "[" + ",".join(str(i) for i in indices) + "]"
    return f"+data.band_indices={idx_str}"


for sensor in SENSOR_YAML_PATHS:
    wavelengths, widths = load_sensor_wavelengths(sensor)
    num_bands = len(wavelengths)
    # Always the TRUE, unpermuted widths for wrong_permuted/wrong_constant/
    # wrong_jitter -- band widths are never corrupted by those three,
    # only center wavelengths are.
    bw_nm_true_order = _dict_list_override("bandwidth_list", sensor, widths)

    for seed in SEEDS:
        rng = random.Random(seed)
        perm = rng.sample(range(num_bands), num_bands)
        wrong_wavelengths = [wavelengths[i] for i in perm]
        condition_name = f"wrong_permuted_s{seed}"
        wl_nm = wave_list_override(sensor, wrong_wavelengths, "nm")
        wl_um = wave_list_override(sensor, wrong_wavelengths, "um")
        print(f"{sensor}\t{condition_name}\t{wl_nm}\t{wl_um}\t{bw_nm_true_order}\t")

        if seed in SHUFFLE_SEEDS:
            # Same permutation as above, but now the pixel DATA is reordered
            # by it too (via band_indices=perm, unsorted) -- so the
            # (data, wavelength, bandwidth) triple at every new position is
            # self-consistent, unlike wrong_permuted's mismatch. See module
            # docstring.
            shuffled_widths = [widths[i] for i in perm]
            bw_nm_shuffled = _dict_list_override("bandwidth_list", sensor, shuffled_widths)
            print(f"{sensor}\tshuffled_s{seed}\t{wl_nm}\t{wl_um}\t{bw_nm_shuffled}\t{band_indices_override(perm)}")

    median_wavelength = statistics.median(wavelengths)
    constant_wavelengths = [median_wavelength] * num_bands
    wl_nm = wave_list_override(sensor, constant_wavelengths, "nm")
    wl_um = wave_list_override(sensor, constant_wavelengths, "um")
    print(f"{sensor}\twrong_constant\t{wl_nm}\t{wl_um}\t{bw_nm_true_order}\t")

    for label, jitter_min_nm, jitter_max_nm, seeds in JITTER_RANGES:
        for seed in seeds:
            rng = random.Random(seed)
            jittered_wavelengths = []
            for w in wavelengths:
                magnitude = rng.uniform(jitter_min_nm, jitter_max_nm)
                sign = rng.choice([-1.0, 1.0])
                jittered_wavelengths.append(w + sign * magnitude)
            condition_name = f"{label}_s{seed}"
            wl_nm = wave_list_override(sensor, jittered_wavelengths, "nm")
            wl_um = wave_list_override(sensor, jittered_wavelengths, "um")
            print(f"{sensor}\t{condition_name}\t{wl_nm}\t{wl_um}\t{bw_nm_true_order}\t")
