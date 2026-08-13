#!/bin/bash
# -----------------------------------------------------------------------------
# OxHyperMinerals (EMIT L2A, multilabel mineral segmentation) with UniverSat-B,
# using the same single-tap ConvSegmentor + UniverSatConvHead path as
# run_universat_convsegmentor.sh -- see that script's header for the full
# rationale (no per-block taps, so UPerNet/MLPFusion are architecturally
# unusable; UniverSatConvHead is a faithful port of UniverSat's own
# LP_eval_conv.py: ConvHead). This script adapts that same wiring for OxHyp's
# different Lightning module (multilabel_segmentation, not segmentation) and
# different sensor (emit, 244 bands, 60m resolution -- not enmap/desis/eo1).
#
# Dataset / normalization / label scheme: see run_oxhyperminerals_upernet.sh's
# header for the full detail (root path fix, split-file verification,
# minerals3ghk 3-class scheme, EMIT bandwise stats computed at
# data/oxhyperminerals_statistics/{mu,sigma}.npy).
#
# patch_size/latent_patch_size: EMIT was never one of UniverSat's own
# pretraining/eval sensors (Table A.3 covers spotRGBN/aerialflair/s2/s1/l7/
# alos/modis/aerial/naip/l8/EO1/GF5/enmap/rgbneon/ndemneon/neon -- no EMIT),
# so there's no published scale to match exactly. Extrapolated from the
# EnMAP-family convention (latent_patch_size=2, i.e. a patch physically
# 2x the sensor's native pixel resolution) for consistency: latent_patch_size=2
# at EMIT's 60m resolution -> 120m physical patches -> a 32x32 latent grid at
# img_size=64 (half resolution in each dim, same ratio as the EnMAP case's
# 64x64 latent at img_size=128). patch_size=1 (full 64x64=4096 dense output
# grid) kept for the same reason as the EnMAP-family script: exercises the
# same "Bilinear_out"/"CA_Sub" recovery mechanism the paper's own EnMAP
# eval used, and empirically costs nothing extra over a coarser grid.
#
# Memory: img_size=64 here vs 128 for the EnMAP-family tasks -- the dominant
# "S1" sub-patch attention bottleneck scales as (img_size/subpatch)^2, so
# this should need ~4x LESS memory than the EnMAP case. batch_size still
# defaults to 4 (the confirmed-safe EnMAP-family value) out of caution rather
# than assuming the 4x headroom means batch_size=8 is safe -- untested at
# this img_size specifically. Bump BATCH_SIZE explicitly to try higher.
#
# Already-completed runs are skipped automatically (checked via a saved
# checkpoint file, not SLURM exit status -- see has_completed_run() below).
# Set FORCE_RESUBMIT=true to resubmit anyway.
#
# Usage:
#   DEBUG_MODE=true bash scripts/downstream/run_oxhyperminerals_universat.sh
#   bash scripts/downstream/run_oxhyperminerals_universat.sh
# -----------------------------------------------------------------------------
set -e

eval "$(micromamba shell hook --shell bash)"
micromamba activate hsi-bench

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
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
TASK="oxhyperminerals_emit_l2a"
SENSOR="emit"
BACKBONE="universat_b"
DECODER="universat_conv_head"
IMG_SIZE=64
PATCH_SIZE=1
LATENT_PATCH_SIZE=2
MAX_EPOCHS="${MAX_EPOCHS:-80}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
BATCH_SIZE="${BATCH_SIZE:-4}"
PROTOCOL="linear"
WEIGHT_INIT="universat_pretrained"
NUM_CLASSES=3

WANDB_PROJECT="hsi-bench"
DISK_LOG_ROOT="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/niklas/experiments/hsi-bench"

SLURM_EXCLUDE='\"hpdar04c04s02,hpdar04c02s05,hpdar04c01s01,hpdar04c01s02,hpdar04c01s03,hpdar04c03s04,hpdar04c03s05,hpdar04c05s03,hpdar04c05s05,hpdar04c06s02,hpdar04c06s01\"'
SLURM_ACCOUNT="${SLURM_ACCOUNT:-hpda-c}"
SLURM_TIMEOUT_MIN="${SLURM_TIMEOUT_MIN:-}"

DEBUG_MODE="${DEBUG_MODE:-false}"
FORCE_RESUBMIT="${FORCE_RESUBMIT:-false}"

EMIT_STATS_PATH="data/oxhyperminerals_statistics"

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
  if [ -f "${UNIVERSAT_LOCAL_REPO}/hubconf.py" ]; then
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

enable_offline_runtime_env() {
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
}

has_completed_run() {
  local experiment_setup="${TASK}_convseg_linear"
  local task_dir="${DISK_LOG_ROOT}/downstream/${experiment_setup}/${SENSOR}/${DECODER}/${BACKBONE}/${PROTOCOL}/${WEIGHT_INIT}"

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

submit_job() {
  local experiment_setup="${TASK}_convseg_linear"

  ensure_offline_assets || {
    echo "Warning: failed to prefetch offline assets for UniverSat. Skipping."
    return 1
  }

  validate_existing_path "EMIT bandwise stats" "$EMIT_STATS_PATH" || return 1

  local task_name="downstream/${experiment_setup}/${SENSOR}/${DECODER}/${BACKBONE}/${PROTOCOL}/${WEIGHT_INIT}"

  local cmd="python src/train.py --config-name=downstream -m"
  cmd+=" experiment=downstream/${TASK}/base"
  cmd+=" data=oxhyperminerals_emit_l2a_10pct"
  cmd+=" backbone@model.backbone_config=${BACKBONE}"
  cmd+=" decoder@model.decoder_config=${DECODER}"
  cmd+=" sensor=${SENSOR}"
  cmd+=" sensor@data.sensor_config=${SENSOR}"
  cmd+=" experiment_setup=${experiment_setup}"
  cmd+=" protocol=${PROTOCOL}"
  cmd+=" weight_init=${WEIGHT_INIT}"
  cmd+=" backbone_name=${BACKBONE}"
  cmd+=" decoder_name=${DECODER}"
  cmd+=" model.model_type=vit_seg"
  cmd+=" model.num_classes=${NUM_CLASSES}"
  cmd+=" model.freeze_backbone=true"
  cmd+=" model.learning_rate=${LEARNING_RATE}"
  cmd+=" model.weight_decay=${WEIGHT_DECAY}"
  cmd+=" model.backbone_config.patch_size=${PATCH_SIZE}"
  cmd+=" model.backbone_config.latent_patch_size=${LATENT_PATCH_SIZE}"
  cmd+=" model.backbone_config.img_size=${IMG_SIZE}"
  cmd+=" model.backbone_config.pretrained=true"
  cmd+=" data.img_size=${IMG_SIZE}"
  cmd+=" data.batch_size=${BATCH_SIZE}"
  cmd+=" trainer=gpu"
  cmd+=" trainer.max_epochs=${MAX_EPOCHS}"
  cmd+=" paths=default"
  cmd+=" paths.log_dir=${DISK_LOG_ROOT}"
  cmd+=" logger.wandb.project=${WANDB_PROJECT}"
  cmd+=" +logger.wandb.name=${TASK}_${BACKBONE}"
  cmd+=" task_name=${task_name}"
  cmd+=" hydra/launcher=slurm_one_gpu"
  cmd+=" hydra.launcher.exclude=${SLURM_EXCLUDE}"
  cmd+=" hydra.launcher.account=${SLURM_ACCOUNT}"
  if [ -n "$SLURM_TIMEOUT_MIN" ]; then
    cmd+=" hydra.launcher.timeout_min=${SLURM_TIMEOUT_MIN}"
  fi
  cmd+=" hydra.sweep.subdir=${BACKBONE}"
  cmd+=" data.apply_input_normalization=false data.standardize=true data.standardization_mode=bandwise data.standardization_stats_path=${EMIT_STATS_PATH}"

  echo ""
  echo "===== Backbone: ${BACKBONE} ====="
  echo "$cmd"

  if [ "$DEBUG_MODE" = "true" ]; then
    return 0
  fi

  mkdir -p "${REPO_ROOT}/scripts/downstream/logs"
  local submit_log="${REPO_ROOT}/scripts/downstream/logs/${TASK}_${BACKBONE}_submit.log"
  ( eval "$cmd" > "${submit_log}" 2>&1 ; echo "[${TASK}/${BACKBONE}] submission client exited with code $?" >> "${submit_log}" ) &
  echo "Backgrounded (client log: ${submit_log})"
  sleep 2
}

enable_offline_runtime_env

if [ "$FORCE_RESUBMIT" != "true" ]; then
  existing_ckpt=$(has_completed_run) && {
    echo ""
    echo "===== Backbone: ${BACKBONE} ====="
    echo "Skipping: found an existing checkpoint from a completed run:"
    echo "  ${existing_ckpt}"
    echo "(set FORCE_RESUBMIT=true to resubmit anyway)"
    exit 0
  }
fi
submit_job || echo "Warning: run was not submitted (see error above)."

echo ""
echo "Job submitted."
