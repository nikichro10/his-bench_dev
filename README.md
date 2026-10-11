# Benchmarking Any-Sensor Foundation Models on Hyperspectral Data

Code for the hyperspectral experiments of the Master's thesis
*Benchmarking Any-Sensor Foundation Models for Earth Observation*
(Niklas Chromik, OTH Regensburg, 2026).

The thesis evaluates sensor-agnostic foundation models, which accept an arbitrary set of spectral
bands together with their wavelength metadata, under a single frozen-encoder protocol. This
repository covers Chapter 5.2: it tests whether these models, pretrained mostly on multispectral
data, still produce useful features for hyperspectral imagery with hundreds of narrow bands, and
how much they rely on the bands and wavelength metadata they receive.

The multispectral and SAR experiments (Chapter 5.1) live in the companion repository
[pangaea-bench-dev-nc](https://github.com/nikichro10/pangaea-bench-dev-nc).
This repository builds on the SpectralEarth-FM codebase, since PANGAEA has no infrastructure
for hyperspectral data.

## Tasks

Nine semantic segmentation tasks across four hyperspectral sensors:

| Task | Sensor | Experiment config |
|---|---|---|
| CDL, EuroCrops, BD-Forêt, TreeMap, BNETD, NLCD | EnMAP | `configs/experiment/downstream/{cdl,eurocrops,bdforet,treemap,bnetd,nlcd}` |
| DESIS-CDL | DESIS | `configs/experiment/downstream/desis_cdl` |
| EO1-CDL | EO-1 Hyperion | `configs/experiment/downstream/eo1_cdl` |
| OxHyperMinerals (10 % split) | EMIT | `configs/experiment/downstream/oxhyperminerals_emit_l2a` |

Split files are in `data/splits/`, normalization statistics in `data/stats/` and per-sensor
wavelength metadata in `configs/sensor/`.

## Models

| Thesis name | Backbone key | Patch size | Notes |
|---|---|---|---|
| DOFA | `dofa_b` | 16 | Hypernetwork |
| DOFA+ | `dofav2_b` | 14 | Hypernetwork |
| Copernicus-FM | `copernicus_fm_b` | 16 | Hypernetwork |
| Panopticon | `panopticon_b` | 14 | Cross-attention |
| CARL | `carl` | 8 | Cross-attention, image-wise normalization only |
| SpecAware | `specaware_b` | 8 | Hyperspectral-specific hypernetwork baseline |
| SpecViT-p8-Frozen / SpecViT-p16-Frozen | `spec_vit_b_p8` / `spec_vit_b_p16` | 8 / 16 | Random init, frozen |
| ViT-p16-Frozen / ViT-p8-Frozen | `vit_b` / `vit_b_p8` | 16 / 8 | Random init, frozen |
| ViT-p16-Sup | `vit_b_finetune` | 16 | Trained end-to-end |

All models use the base size and an input size of 224 px, with the same patch sizes as in the
multispectral experiments. Checkpoints are expected under
`pretrained_models/comparison/{dofa_b,dofav2_b,copernicus_fm_b,panopticon,carl,specaware}/`.
The header of `scripts/downstream/run_pangaea_consistency_cdl.sh` lists the download sources and
how each backbone's weights are loaded.

## Setup

```bash
micromamba create -n hsi-bench python=3.11
micromamba activate hsi-bench
pip install -r requirements.txt
```

Cluster path profiles are selected with `paths=<profile>` (`configs/paths/default.yaml`,
`juwels.yaml`, `jureca.yaml`).

## Running an experiment

All runs go through `src/train.py` with the `downstream` Hydra config. The launch scripts build the
full command per task and backbone, including backbone-specific weight loading, normalization and
decoder taps, and submit it to SLURM via the submitit launcher:

```bash
# all tasks and backbones
bash scripts/downstream/run_pangaea_consistency_mlp_fusion.sh

# a subset
TASKS_CSV=eurocrops BACKBONES_CSV=carl,panopticon_b bash scripts/downstream/run_pangaea_consistency_mlp_fusion.sh
```

Runs that already have a saved checkpoint are skipped; set `FORCE_RESUBMIT=true` to rerun them.

Protocol used in the thesis: frozen encoder, MLPFusion decoder, AdamW (lr 1e-4, weight decay 0.05),
cosine annealing learning rate schedule, batch size 8, 80 epochs, checkpoint with the lowest
validation loss used for testing. This differs from the PANGAEA protocol of the multispectral
experiments in two points: PANGAEA decays the learning rate in two steps and selects the
checkpoint by validation mIoU.

## Reproducing the thesis results

| Thesis | Scripts (`scripts/downstream/`) | Analysis (`notebooks/`) |
|---|---|---|
| Table 13: main results (EnMAP, DESIS, EO-1 tasks) | `run_pangaea_consistency_mlp_fusion.sh` | `benchmark_results.ipynb` |
| Table 13: main results (OxHyperMinerals) | `run_oxhyperminerals_mlpfusion.sh` | `benchmark_results.ipynb` |
| Figure 20: random band subsets (3 to 20 bands vs. all) | `run_vit_s_small_band_ablation.sh` (ViT-S draw selection), `run_small_band_ablation.sh` | `small_band_ablation_results.ipynb` → `figures/*_band_count_bars.pdf` |
| Figure 21: corrupted wavelength metadata (jitter, permutation, constant) | `run_wrong_wavelength_ablation.sh` | `wrong_wavelength_ablation_results.ipynb` → `figures/*_wrong_wavelength_bars.pdf` |

Figures 20 and 21 cover OxHyperMinerals, EuroCrops and BD-Forêt. The band subsets and corrupted
wavelengths passed to each run are printed by `print_small_band_overrides.py` and
`print_wrong_wavelength_overrides.py`.

The notebooks read results from the experiment root
`/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/niklas/experiments/hsi-bench`. The scripts contain the
cluster paths and SLURM settings of the LRZ Terrabyte cluster used for the thesis; adjust
`DISK_LOG_ROOT`, the path profile and the SLURM account for another environment.

`run_pangaea_consistency_cdl.sh` and `run_oxhyperminerals_upernet.sh` run the same benchmark with the
UPerNet decoder. These runs are not reported in the thesis.

## Other code in this repo

- `scripts/downstream/additional_experiments/` and `notebooks/additional_experiments/` hold further
  exploratory ablations (semantic vs. random band groups, UniverSat-B variants, hypernetwork
  re-initialization, UNet baseline) that are not part of the thesis results.
- `scripts/pretrain/` and the SpectralEarth-FM / Hiera multisensor backbone code belong to a separate
  project (self-supervised pretraining of a different model family) and are not used for this benchmark.

## Acknowledgements

This code builds on the SpectralEarth-FM codebase. The comparison backbones are adapted from the
original implementations of the respective models. The experiments were run on the LRZ Terrabyte cluster.
