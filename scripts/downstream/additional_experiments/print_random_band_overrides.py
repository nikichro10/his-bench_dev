"""Print Hydra CLI overrides for random band-subset conditions used by
run_random_band_ablation.sh -- the counterpart to
print_band_group_overrides.py's semantic groups (rgb/vis/vnir/swir).

Answers a different question than the semantic-group experiment: does a
domain-meaningful contiguous region actually beat an arbitrary same-size
random subset, or does band COUNT alone explain the earlier results? For
each band count in SIZE_LABELS, draws 3 independent random subsets (fixed
seeds 0/1/2, uniform without replacement across the full spectrum) per
sensor.

Deliberately avoids importing src.* for the same reason as
print_band_group_overrides.py (torch/lightning import chain can hang on the
login node outside a SLURM allocation) -- reads sensor YAML/CSVs directly
with plain yaml/csv, duplicating the small amount of shared loading logic
rather than importing that script as a module (which would also re-run its
own top-level print loop as a side effect).

Size labels "rgb", "vis", "vnir", "swir" name which semantic group's band
COUNT is being matched for that sensor -- NOT the semantic group's actual
bands. E.g. "vnir" here means "as many random bands as EMIT's vnir group has
(84)", drawn from anywhere in the spectrum, not from the vnir wavelength
range itself. "n20" is not matched to any semantic group -- it's a fixed
20-band count added as a midpoint between rgb (3) and vis (43/53), purely to
fill in the count-vs-performance curve; it has no semantic-region
counterpart, so it only ever appears in the random-subset ablation, never in
Part 1's semantic-region charts.

Output: one TAB-separated line per (sensor, size_label, seed) combo:
    <sensor>\\t<condition_name>\\t<band_count>\\t<common overrides>\\t<wave_list override, nm>\\t<wave_list override, um>\\t<bandwidth_list override, nm>

condition_name is e.g. "rand_vnir_s1" -- used as the output-path/wandb-name
segment, parallel to run_wavelength_ablation.sh's "${group}" segment.
band_count is the same number embedded in the "common" override's
model.probe_in_chans -- exposed as its own field so callers that need it
directly (e.g. run_vit_b_band_ablation.sh's model.backbone_config.in_chans
override -- vit_b has no wave_list, so this is the only way it learns its
input channel count) don't have to parse it back out of that string.

Quoting rationale for wave_list/bandwidth_list overrides is identical to
print_band_group_overrides.py -- see that script's header.
"""

import csv
import random
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

# Order matters only for readability of the output -- n20 sits between rgb
# and vis since that's the gap it's filling.
SIZE_LABELS = ["rgb", "n20", "vis", "vnir", "swir"]
FIXED_SIZE_COUNTS = {"n20": 20}
SEEDS = [0, 1, 2]


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


def group_size(size_label: str, rgb_indices: list[int], wavelengths: list[float]) -> int:
    """Band count for this size_label -- either a fixed count (n20) or the
    real semantic group's count (rgb/vis/vnir/swir)."""
    if size_label in FIXED_SIZE_COUNTS:
        return FIXED_SIZE_COUNTS[size_label]
    if size_label == "rgb":
        return len(rgb_indices)
    lo, hi = BAND_GROUP_RANGES_NM[size_label]
    return sum(1 for w in wavelengths if lo <= w < hi)


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
    num_bands = len(wavelengths)
    for size_label in SIZE_LABELS:
        n = group_size(size_label, rgb_indices, wavelengths)
        for seed in SEEDS:
            rng = random.Random(seed)
            indices = sorted(rng.sample(range(num_bands), n))
            group_wavelengths = [wavelengths[i] for i in indices]
            group_widths = [widths[i] for i in indices]

            condition_name = f"rand_{size_label}_s{seed}"
            idx_str = "[" + ",".join(str(i) for i in indices) + "]"
            common = f"+data.band_indices={idx_str} model.probe_in_chans={len(indices)}"
            wl_nm = wave_list_override(sensor, group_wavelengths, "nm")
            wl_um = wave_list_override(sensor, group_wavelengths, "um")
            bw_nm = _dict_list_override("bandwidth_list", sensor, group_widths)
            print(f"{sensor}\t{condition_name}\t{len(indices)}\t{common}\t{wl_nm}\t{wl_um}\t{bw_nm}")
