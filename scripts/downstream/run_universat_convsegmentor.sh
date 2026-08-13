#!/bin/bash
# -----------------------------------------------------------------------------
# UniverSat-B (https://github.com/gastruc/UniverSat) on the same 8 downstream
# tasks as run_pangaea_consistency_cdl.sh / run_pangaea_consistency_mlp_fusion.sh,
# same hyperparameters (AdamW, lr=1e-4, weight_decay=0.05, batch_size=8,
# 80 epochs, frozen linear probe) -- but with a DIFFERENT decoder pairing,
# because UniverSat is architecturally incompatible with the other two:
#
#   UniverSat's model.encode(...) returns a single dense feature grid at a
#   caller-chosen output resolution -- there is no way to pull per-block
#   intermediate features the way UPerNet/MLPFusion need (both require 4 tap
#   layers, e.g. [3,5,7,11]). So this script uses his-bench's single-tap path
#   instead: model_type=vit_seg (ConvSegmentor) + decoder=conv_head. Because
#   of this, UniverSat's results are NOT directly comparable to the
#   UPerNet/MLPFusion columns for the other 11 encoders in the benchmark
#   notebook -- they belong in their own table.
#
# Tasks: cdl, bdforet, bnetd, eurocrops, nlcd, treemap (all sensor=enmap),
# plus desis_cdl (sensor=desis) and eo1_cdl (sensor=eo1).
#
# Sensor coverage / normalization, verified before wiring this in:
#   - EnMAP: UniverSat's own modality_registry.py registers "enmap" with 202
#     bands and wavelengths that match his-bench's own EnMAP metadata almost
#     exactly (0.418-2.445 um both sides).
#   - EO1: UniverSat's registry has "EO1" at 175 bands, but his-bench's own
#     EO1 processing is 163 bands -- a real mismatch, not just a naming
#     difference. DESIS isn't registered at all.
#   - Src/backbones/universat_encoder.py therefore NEVER relies on
#     UniverSat's built-in registry lookup -- it always passes explicit
#     wavelengths/input_res/subpatches from his-bench's own SensorRegistry,
#     for enmap too (not just eo1/desis), so every sensor goes through the
#     identical code path. Verified safe by reading UniversalPatchEncoder's
#     source: the modality *name* string is never used to index a learned
#     per-sensor embedding -- only the wavelength values are (numeric
#     wavelengths go through Learnable Fourier Features; only a small closed
#     set of SAR/DEM channel *codes* route to fixed learned parameters, which
#     hyperspectral sensors never use). An unseen name like "desis" is
#     therefore safe.
#   - Normalization: UniverSat's own EnMAP training/eval pipeline
#     (src/data/EnmapDataset.py in their repo) applies bandwise z-score
#     standardization from a precomputed per-band mean/std file -- the same
#     mechanism (not the same numbers) already used here for the DOFA-family
#     and Panopticon-B (data.standardize=true
#     data.standardization_mode=bandwise). UniverSat's own stats file is
#     tied to their specific SpectralEarthEval corpus copy (auto-computed,
#     not published as fixed constants), so this script reuses his-bench's
#     own already-computed dofa_family_stats_path_map directories instead --
#     same transform, different (but compatible) corpus, consistent with how
#     every other backbone here reuses those same stats directories rather
#     than re-deriving paper-original numbers.
#   - patch_size/latent_patch_size (src/backbones/universat_encoder.py) are
#     DECOUPLED, faithful to UniverSat's own EnMAP eval config
#     (EnmapCdl.yaml in their repo: scale_base=6, num_patches=16384=128^2,
#     out_patch_size=1):
#       * latent_patch_size=2 controls the model's own INPUT tokenization
#         scale (patch_size in metres = latent_patch_size * sensor
#         resolution_m = 2*30 = 60m -> scale_base=6, matching their config
#         exactly) -> a 64x64 latent grid at img_size=128.
#       * patch_size=1 controls the OUTPUT grid requested from encode() (and
#         what ConvSegmentor reshapes into): img_size/patch_size = 128 ->
#         full 128x128=16384-token DENSE per-pixel output, exercising
#         UniverSat's "Bilinear_out"/"CA_Sub" sub-patch-recovery mechanism,
#         matching out_patch_size=1 exactly. An earlier version of this
#         script set patch_size=8 for both input and output (a coarse
#         16x16=256-token grid matched to the rest of this benchmark's
#         convention, avoiding the recovery mechanism entirely) -- NOT
#         faithful to the paper; superseded once real GPU tests showed the
#         dense output costs nothing extra (see CAUTION below).
#
# Weight loading: model.backbone_config.pretrained=true, loaded via
# UniverSat's own HuggingFace-Hub from_pretrained("g-astruc/UniverSat")
# inside src/backbones/universat_encoder.py -- NOT a local checkpoint file
# (unlike every other pretrained backbone in this benchmark). Requires the
# UniverSat repo cloned locally (this script does that automatically, see
# ensure_offline_assets) so torch.hub can load hubconf.py's model-building
# code, and the HF weights prefetched into the shared offline cache before
# HF_HUB_OFFLINE=1 is set for the actual (usually internet-less) SLURM job.
#
# CAUTION -- memory-constrained, learned the hard way via four real OOMs:
# the actual bottleneck (confirmed from OOM tracebacks landing in
# utils_ViT.py's fc1, inside UniversalPatchEncoder's "S1" sub-patch
# axial-attention block) scales as (img_size/subpatch)^2 -- independent of
# how img_size factors into patch_size x latent_patch_size, and (verified
# empirically) independent of the OUTPUT grid size too: batch_size=8 OOMs
# identically ("Tried to allocate 37.88 GiB") whether the output grid is the
# coarse 16x16 or the full dense 128x128 -- so switching to the dense,
# paper-faithful output grid cost nothing extra. subpatch=1 for
# enmap/eo1/desis is what UniverSat's own paper (Table A.3) actually used,
# so it's kept as-is rather than traded for a bigger batch size. batch_size=4
# was confirmed via real GPU jobs to run cleanly end-to-end (train+val+test,
# real checkpoint saved) at img_size=128 with the full dense output grid --
# 4x the throughput of batch_size=1 at zero fidelity cost. Recommend
# submitting a single task first (TASKS_CSV=cdl) and checking its .submitit
# log / wandb train_loss closely before submitting the rest, in case a given
# task's data pipeline pushes memory differently.
#
# Already-completed (task) combinations are skipped automatically (checked
# via a saved checkpoint file, not SLURM exit status -- see has_completed_run()
# below). Set FORCE_RESUBMIT=true to resubmit anyway.
#
# Usage:
#   TASKS_CSV=cdl DEBUG_MODE=true bash scripts/downstream/run_universat_convsegmentor.sh
#   TASKS_CSV=cdl bash scripts/downstream/run_universat_convsegmentor.sh
#   bash scripts/downstream/run_universat_convsegmentor.sh
# -----------------------------------------------------------------------------
set -e

# Activate the hsi-bench micromamba env (created via:
#   micromamba create -n hsi-bench python=3.11
#   micromamba run -n hsi-bench pip install -r requirements.txt
# ) so hydra/submitit are available to the submitting process itself, not
# just inside the eventual SLURM job.
eval "$(micromamba shell hook --shell bash)"
micromamba activate hsi-bench

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
# Confirmed via real test (batch_size=8 still OOM'd even with this set, but
# with fragmentation reduced from 11.5 GiB reserved-unallocated to ~0) --
# doesn't fix batch_size=8, but costs nothing and helps at batch_size=4 too.
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO_ROOT="$(pwd -P)"
OFFLINE_ASSET_ROOT="${OFFLINE_ASSET_ROOT:-${REPO_ROOT}/pretrained_models/comparison/offline_assets}"
HF_HOME_SHARED="${HF_HOME_SHARED:-${OFFLINE_ASSET_ROOT}/hf_home}"
HF_HUB_CACHE_SHARED="${HF_HUB_CACHE_SHARED:-${HF_HOME_SHARED}/hub}"
TRANSFORMERS_CACHE_SHARED="${TRANSFORMERS_CACHE_SHARED:-${HF_HOME_SHARED}/transformers}"
TORCH_HOME_SHARED="${TORCH_HOME_SHARED:-${OFFLINE_ASSET_ROOT}/torch_home}"
UNIVERSAT_LOCAL_REPO="${UNIVERSAT_LOCAL_REPO:-${OFFLINE_ASSET_ROOT}/universat_repo}"
UNIVERSAT_PREFETCH_MARKER="${OFFLINE_ASSET_ROOT}/.universat_prefetched"

mkdir -p "${HF_HUB_CACHE_SHARED}" "${TRANSFORMERS_CACHE_SHARED}" "${TORCH_HOME_SHARED}"
export HF_HOME="${HF_HOME_SHARED}"
export HF_HUB_CACHE="${HF_HUB_CACHE_SHARED}"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE_SHARED}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE_SHARED}"
export TORCH_HOME="${TORCH_HOME_SHARED}"

# --- Fixed experiment configuration -----------------------------------------
TASKS=(cdl bdforet bnetd eurocrops nlcd treemap desis_cdl eo1_cdl)
TASKS_CSV="${TASKS_CSV:-}"
if [ -n "$TASKS_CSV" ]; then
  IFS=',' read -ra TASKS <<< "$TASKS_CSV"
fi

declare -A task_sensor_map=(
  ["cdl"]="enmap"
  ["bdforet"]="enmap"
  ["bnetd"]="enmap"
  ["eurocrops"]="enmap"
  ["nlcd"]="enmap"
  ["treemap"]="enmap"
  ["desis_cdl"]="desis"
  ["eo1_cdl"]="eo1"
)

BACKBONE="universat_b"
# NOT src.decoders.conv_head.ConvHead (this repo's generic single-tap head,
# which does multiple internal upsample+conv steps to reach full resolution).
# UniverSat's own paper/repo (src/LP_eval_conv.py: ConvHead) used a different,
# simpler head for its EnMAP evaluation: LayerNorm -> Conv2d(3x3)+BN+ReLU ->
# Conv2d(1x1), staying at token-grid resolution with no internal upsampling
# -- their own eval loop bilinear-interpolates the output to label resolution
# afterward. Ported faithfully as UniverSatConvHead
# (src/decoders/universat_conv_head.py); ConvSegmentor's existing
# _match_input_spatial_size already does the same external bilinear
# interpolate step for any decoder whose output doesn't match the input size,
# so no changes to ConvSegmentor itself were needed.
DECODER="universat_conv_head"
# NOT 224 (the img_size used by every other backbone in this benchmark).
# UniverSat's per-pixel Learnable-Fourier-Features patch embedding runs
# BEFORE any spatial/spectral token collapsing, so its memory scales with
# raw H*W*C directly -- confirmed by a real OOM at img_size=224/batch_size=8
# ("Tried to allocate 58.00 GiB" on an 80GB GPU). 128 matches UniverSat's own
# published EnMAP eval configs (configs/dataset/EnmapCdl.yaml etc. in their
# repo: height: 128, width: 128) and cuts per-pixel cost to ~33% of 224x224 --
# comfortably fits. This means UniverSat's results are at a different input
# resolution than the rest of the benchmark, on top of the already-different
# decoder -- another reason its table isn't directly comparable to the other
# 11 encoders' UPerNet/MLPFusion columns.
IMG_SIZE=128
# patch_size=1 -> full dense 128x128 output grid; latent_patch_size=2 ->
# 60m/30m=2px/token input tokenization (scale_base=6). See header comment.
PATCH_SIZE=1
LATENT_PATCH_SIZE=2
MAX_EPOCHS="${MAX_EPOCHS:-80}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
# NOT 8 (every other backbone in this benchmark). The true bottleneck is the
# "S1" sub-patch axial-attention block inside UniverSat's UniversalPatchEncoder
# (confirmed from a real OOM traceback landing in utils_ViT.py's fc1 layer,
# not the initial per-pixel embedding as first suspected) -- its cost scales
# as (img_size/subpatch)^2, and subpatch=1 for enmap/eo1/desis is what
# UniverSat's own paper (Table A.3) actually used, so it's kept as-is rather
# than traded away for a bigger batch size. batch_size=8 still OOMs at
# img_size=128 even with PYTORCH_ALLOC_CONF=expandable_segments:True (which
# did eliminate fragmentation -- 81 MiB reserved-unallocated vs 11.5 GiB
# before -- but the real allocation genuinely doesn't fit: tried to allocate
# 37.88 GiB against 63.36 GiB already in use on an 80GB GPU). batch_size=4
# was confirmed via a real GPU job to run cleanly end-to-end (train+val+test,
# real checkpoint saved) at img_size=128, subpatches=1 -- 4x the throughput
# of batch_size=1 without touching subpatch fidelity.
BATCH_SIZE="${BATCH_SIZE:-4}"
PROTOCOL="linear"
WEIGHT_INIT="universat_pretrained"

WANDB_PROJECT="hsi-bench"
DISK_LOG_ROOT="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/niklas/experiments/hsi-bench"

# hpdar04c06s01 added after 3 straight jobs on it failed with
# "CUDA unknown error"/"No supported gpu backend found" -- confirmed via a
# direct `srun --nodelist=hpdar04c06s01 ... torch.cuda.is_available()` probe
# that the node's GPU is genuinely broken right now, unrelated to this
# script or UniverSat.
SLURM_EXCLUDE='\"hpdar04c04s02,hpdar04c02s05,hpdar04c01s01,hpdar04c01s02,hpdar04c01s03,hpdar04c03s04,hpdar04c03s05,hpdar04c05s03,hpdar04c05s05,hpdar04c06s02,hpdar04c06s01\"'
SLURM_ACCOUNT="${SLURM_ACCOUNT:-hpda-c}"
# UniverSat (~201M params, axial cross-attention) is heavier than most other
# backbones in this benchmark; NLCD is already the heaviest task. Empty by
# default (keep the launcher's own 600min); bump if a run times out, e.g.
# SLURM_TIMEOUT_MIN=1200 TASKS_CSV=nlcd bash scripts/downstream/run_universat_convsegmentor.sh
SLURM_TIMEOUT_MIN="${SLURM_TIMEOUT_MIN:-}"

DEBUG_MODE="${DEBUG_MODE:-false}"
FORCE_RESUBMIT="${FORCE_RESUBMIT:-false}"

# Same DOFA-family bandwise stats used elsewhere in this benchmark -- see
# header comment for why UniverSat reuses these rather than paper-original
# (unpublished, corpus-tied) numbers.
declare -A dofa_family_stats_path_map=(
  ["enmap"]="data/dofa_statistics"
  ["desis"]="data/desis_statistics"
  ["eo1"]="data/eo1_statistics"
)

# --- Helpers -----------------------------------------------------------------

validate_existing_path() {
  local label="$1" path="$2"
  if [ ! -e "$path" ]; then
    echo "Error: ${label} path does not exist: ${path}" >&2
    return 1
  fi
}

prefetch_universat_weights() {
  echo "Prefetching UniverSat weights from HuggingFace Hub (g-astruc/UniverSat)..."
  HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python3 - "$UNIVERSAT_LOCAL_REPO" <<'PYEOF'
import sys
import torch
repo = sys.argv[1]
model = torch.hub.load(repo, "from_pretrained", repo_id="g-astruc/UniverSat", compile=False, source="local")
del model
PYEOF
}

ensure_universat_repo() {
  if [ -d "${UNIVERSAT_LOCAL_REPO}/hubconf.py" ] || [ -f "${UNIVERSAT_LOCAL_REPO}/hubconf.py" ]; then
    echo "Using existing UniverSat repo cache at ${UNIVERSAT_LOCAL_REPO}"
    return 0
  fi
  echo "Cloning UniverSat repo into ${UNIVERSAT_LOCAL_REPO}"
  mkdir -p "$(dirname "${UNIVERSAT_LOCAL_REPO}")"
  git clone --depth 1 https://github.com/gastruc/UniverSat.git "${UNIVERSAT_LOCAL_REPO}"
}

ensure_offline_assets() {
  ensure_universat_repo || return 1
  if [ ! -f "${UNIVERSAT_PREFETCH_MARKER}" ]; then
    prefetch_universat_weights || return 1
    touch "${UNIVERSAT_PREFETCH_MARKER}"
  fi
}

build_norm_overrides() {
  local sensor="$1"
  local stats_path="${dofa_family_stats_path_map[$sensor]}"
  if [ -z "$stats_path" ]; then
    echo "Error: no bandwise stats path configured for sensor '${sensor}'." >&2
    return 1
  fi
  validate_existing_path "bandwise stats (${sensor})" "$stats_path" || return 1
  echo "data.apply_input_normalization=false data.standardize=true data.standardization_mode=bandwise data.standardization_stats_path=${stats_path}"
}

enable_offline_runtime_env() {
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
}

has_completed_run() {
  local task="$1"
  local sensor="${task_sensor_map[$task]}"
  local experiment_setup="${task}_convseg_linear"
  local task_dir="${DISK_LOG_ROOT}/downstream/${experiment_setup}/${sensor}/${DECODER}/${BACKBONE}/${PROTOCOL}/${WEIGHT_INIT}"

  # See run_pangaea_consistency_cdl.sh's has_completed_run for why this
  # checks the "Best ckpt path:" log line rather than checkpoint existence
  # or sacct/squeue state.
  local log_line ckpt
  log_line=$(grep -h "Best ckpt path:" "${task_dir}"/multiruns/*/.submitit/*/*_0_log.out 2>/dev/null | tail -1)
  if [ -n "$log_line" ]; then
    ckpt="${log_line#*Best ckpt path: }"
    echo "$ckpt"
    return 0
  fi
  return 1
}

# --- Job submission ------------------------------------------------------------

submit_task() {
  local task="$1"
  local sensor="${task_sensor_map[$task]}"
  local experiment_setup="${task}_convseg_linear"

  ensure_offline_assets || {
    echo "Warning: failed to prefetch offline assets for UniverSat. Skipping."
    return 1
  }

  local norm_overrides
  norm_overrides=$(build_norm_overrides "$sensor") || return 1

  local task_name="downstream/${experiment_setup}/${sensor}/${DECODER}/${BACKBONE}/${PROTOCOL}/${WEIGHT_INIT}"

  local cmd="python src/train.py --config-name=downstream -m"
  cmd+=" experiment=downstream/${task}/base"
  cmd+=" backbone@model.backbone_config=${BACKBONE}"
  cmd+=" decoder@model.decoder_config=${DECODER}"
  cmd+=" sensor=${sensor}"
  cmd+=" sensor@data.sensor_config=${sensor}"
  cmd+=" experiment_setup=${experiment_setup}"
  cmd+=" protocol=${PROTOCOL}"
  cmd+=" weight_init=${WEIGHT_INIT}"
  cmd+=" backbone_name=${BACKBONE}"
  cmd+=" decoder_name=${DECODER}"
  cmd+=" model.model_type=vit_seg"
  cmd+=" model.freeze_backbone=true"
  cmd+=" model.learning_rate=${LEARNING_RATE}"
  cmd+=" model.weight_decay=${WEIGHT_DECAY}"
  cmd+=" model.backbone_config.patch_size=${PATCH_SIZE}"
  cmd+=" model.backbone_config.latent_patch_size=${LATENT_PATCH_SIZE}"
  cmd+=" model.backbone_config.img_size=${IMG_SIZE}"
  cmd+=" model.backbone_config.pretrained=true"
  cmd+=" data.img_size=${IMG_SIZE}"
  cmd+=" data.batch_size=${BATCH_SIZE}"
  # See run_pangaea_consistency_cdl.sh's comment on the same override: base
  # experiment configs default to trainer=ddp (devices: 4), but
  # slurm_one_gpu only allocates 1 GPU per job.
  cmd+=" trainer=gpu"
  cmd+=" trainer.max_epochs=${MAX_EPOCHS}"
  cmd+=" paths=default"
  cmd+=" paths.log_dir=${DISK_LOG_ROOT}"
  cmd+=" logger.wandb.project=${WANDB_PROJECT}"
  cmd+=" +logger.wandb.name=${task}_${BACKBONE}"
  cmd+=" task_name=${task_name}"
  cmd+=" hydra/launcher=slurm_one_gpu"
  cmd+=" hydra.launcher.exclude=${SLURM_EXCLUDE}"
  cmd+=" hydra.launcher.account=${SLURM_ACCOUNT}"
  if [ -n "$SLURM_TIMEOUT_MIN" ]; then
    cmd+=" hydra.launcher.timeout_min=${SLURM_TIMEOUT_MIN}"
  fi
  cmd+=" hydra.sweep.subdir=${BACKBONE}"
  cmd+=" ${norm_overrides}"

  echo ""
  echo "===== Task: ${task} | Backbone: ${BACKBONE} ====="
  echo "$cmd"

  if [ "$DEBUG_MODE" = "true" ]; then
    return 0
  fi

  mkdir -p "${REPO_ROOT}/scripts/downstream/logs"
  local submit_log="${REPO_ROOT}/scripts/downstream/logs/${task}_${BACKBONE}_submit.log"
  ( eval "$cmd" > "${submit_log}" 2>&1 ; echo "[${task}/${BACKBONE}] submission client exited with code $?" >> "${submit_log}" ) &
  echo "Backgrounded (client log: ${submit_log})"
  sleep 2
}

enable_offline_runtime_env

for task in "${TASKS[@]}"; do
  if [ "$FORCE_RESUBMIT" != "true" ]; then
    existing_ckpt=$(has_completed_run "$task") && {
      echo ""
      echo "===== Task: ${task} | Backbone: ${BACKBONE} ====="
      echo "Skipping: found an existing checkpoint from a completed run:"
      echo "  ${existing_ckpt}"
      echo "(set FORCE_RESUBMIT=true to resubmit anyway)"
      continue
    }
  fi
  submit_task "$task" || echo "Warning: ${task}/${BACKBONE} run was not submitted (see error above)."
done

echo ""
echo "All jobs submitted."
