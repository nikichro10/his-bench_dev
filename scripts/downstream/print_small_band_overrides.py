"""Print Hydra CLI overrides for the small-band-count random-subset sweep
used by run_vit_s_small_band_ablation.sh (Phase 1) and, contingent on those
results, later by a pretrained-encoder counterpart script (Phase 2).

Different sweep than print_random_band_overrides.py's SIZE_LABELS
(rgb/n20/vis/vnir/swir, matched to the semantic-group band counts). This one
is a clean linear sweep in the SMALL-band regime -- 3, 7, 10, 15, 20 bands --
purpose-built to answer "does test performance climb as we hand a supervised,
fully-finetuned ViT-Small more and more random bands", i.e. is there usable
information in these small subsets at all, before spending compute re-running
them through the frozen-linear-probe pretrained encoders.

n=3 and n=20 draws are numerically IDENTICAL to print_random_band_overrides.py's
existing "rgb" and "n20" draws -- random.Random(seed).sample(range(num_bands), n)
depends only on (seed, n), not on which script or label asked for it. Kept as
separate "n3"/"n20" labels here (not reusing the "rgb" name) so this sweep
reads as a clean, self-contained band-COUNT progression, with no implied tie
to any semantic region -- readers of this experiment's notebook shouldn't
have to know that "n3" happens to coincide with the "rgb" condition from a
different experiment to understand it.

Deliberately avoids importing src.* -- same reason as
print_band_group_overrides.py (torch/lightning import chain can hang on the
login node outside a SLURM allocation).

Sensors: emit + enmap (covers all 4 Phase-1 tasks -- oxhyperminerals_emit_l2a
is emit; eurocrops/bnetd/bdforet are all enmap).

Output: one TAB-separated line per (sensor, size_label, seed) combo:
    <sensor>\\t<condition_name>\\t<band_count>\\t<common overrides>\\t<wave_list override, nm>\\t<wave_list override, um>\\t<bandwidth_list override, nm>

wave_list/bandwidth_list are computed and emitted even though Phase 1's
vit_s has no wave_list field at all (ignores those columns) -- so the SAME
generator script can serve a Phase 2 pretrained-encoder script later without
changes, mirroring print_random_band_overrides.py's shape exactly. Quoting
rationale for wave_list/bandwidth_list is identical to
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

SIZE_LABELS = ["n3", "n7", "n10", "n15", "n20"]
FIXED_SIZE_COUNTS = {"n3": 3, "n7": 7, "n10": 10, "n15": 15, "n20": 20}
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
    for size_label in SIZE_LABELS:
        n = FIXED_SIZE_COUNTS[size_label]
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
