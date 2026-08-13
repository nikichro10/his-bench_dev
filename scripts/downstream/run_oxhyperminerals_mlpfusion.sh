#!/bin/bash
# -----------------------------------------------------------------------------
# OxHyperMinerals (EMIT L2A, multilabel mineral segmentation) with the
# MLPFusionHead decoder (bilinear upsample_mode), across every "normal"
# (non-UniverSat) backbone in this benchmark -- same task as
# run_oxhyperminerals_upernet.sh, mirrors run_pangaea_consistency_mlp_fusion.sh's
# structure instead: model_type=multi_tap_seg (MultiTapSegmentor), tap
# indices forced per backbone (build_tap_overrides below) to match exactly
# what the UPerNet script used, so decoder choice is the only variable
# between the two OxHyp scripts' results for a given backbone.
#
# Adapted for a single task with its own sensor (emit) and a different Lightning module
# (multilabel_segmentation, not segmentation): targets are per-pixel
# multi-hot mineral vectors (a pixel can be e.g. both Goethite and Hematite
# simultaneously), not a single mutually-exclusive class, so loss/metrics are
# BCE + multilabel Accuracy/F1/IoU instead of cross-entropy + MulticlassIoU.
#
# Dataset: /dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/oxhyperminerals_emit_l2a_author64_valid99
# (73,259 64x64 EMIT L2A tiles, pre-split train=48792/val=12094/test=12373 --
# confirmed the checked-in data/splits/oxhyperminerals_emit_l2a/{train,val,test}.txt
# match this data exactly, same counts, sample filenames present on disk).
# configs/data/oxhyperminerals_emit_l2a.yaml's `root` was pointing at a stale
# ${paths.data_dir}-relative location; fixed to the absolute path above.
#
# Label scheme: minerals3ghk (3 classes: Goethite, Hematite, Kaolinite) --
# the DEFAULT data config, not the 10-class minerals10 variant, to keep this
# first pass tractable; num_classes=3 accordingly. Uses data=oxhyperminerals_emit_l2a_10pct
# -- a stratified random 10% PATCH-level subsample (not scene-level) of the
# full split (train=4879/val=1209/test=1237, seed=42, stratified to preserve
# the full split's positive-label fraction; see
# data/splits/oxhyperminerals_emit_l2a/random_patch_f0p1_minerals3ghk_seed42/subset_summary.json).
# An earlier version of this script used the full dataset
# (data=oxhyperminerals_emit_l2a) -- switched to 10pct at the user's request;
# those full-dataset runs were cancelled and their result directories wiped
# before resubmitting, since the output path doesn't otherwise distinguish
# which data variant produced a given checkpoint.
#
# is_mlp_fusion decoder-override detection was MISSING from
# src/models/multilabel_segmentation_module.py (present in
# semantic_segmentation_module.py, but never ported here) -- would have
# broken MLPFusionHead (expects in_channels_list, not num_input_features).
# Fixed by adding the same is_mlp_fusion branch that
# semantic_segmentation_module.py already has. UPerNet itself was unaffected
# (already had its own is_upernet branch) -- verified both decoders via a
# real Hydra instantiate + forward pass before wiring this script in.
#
# Normalization: DOFA-family/Panopticon-B need bandwise z-score stats, same
# as every other task, but no EMIT-specific stats file existed anywhere in
# this repo (unlike enmap/desis/eo1). Computed one from a ~6400-tile sample
# of the real training set (data/oxhyperminerals_statistics/{mu,sigma}.npy)
# using the same apply_input_normalization=false/standardize=false raw-value
# convention the other stats files use, masking invalid_value=-9999 pixels.
#
# UniverSat is handled by a SEPARATE script (run_oxhyperminerals_universat.sh)
# -- same reasons as the rest of this benchmark: single dense-feature-grid
# output, no per-block taps, needs the single-tap ConvSegmentor path instead.
#
# IMG_SIZE=224, NOT OxHyp's native 64px tile size. All 11 backbones run at
# 224 like every other task in this benchmark: BaseSegmentationDataModule
# applies a Kornia K.Resize(img_size) inside on_after_batch_transfer, a
# Lightning hook that only fires when a real Trainer drives the batch
# (confirmed directly -- calling the dataloader standalone bypasses it, which
# is what caused an earlier, WRONG conclusion that img_size doesn't resize
# OxHyp tiles at all and led to excluding dofa_b/dofav2_b/specaware_b here;
# re-tested through an actual Trainer + on_after_batch_transfer call and all
# three load and run cleanly at 224 -- same mechanism enmap/eo1/desis rely on
# for their own native 128px tiles). No backbone is excluded for this task.
#
# Already-completed backbones are skipped automatically (checked via a saved
# checkpoint file, not SLURM exit status -- see has_completed_run() below).
# Set FORCE_RESUBMIT=true to resubmit anyway.
#
# Usage:
#   BACKBONES_CSV=vit_b DEBUG_MODE=true bash scripts/downstream/run_oxhyperminerals_mlpfusion.sh
#   bash scripts/downstream/run_oxhyperminerals_mlpfusion.sh
# -----------------------------------------------------------------------------
set -e

eval "$(micromamba shell hook --shell bash)"
micromamba activate hsi-bench

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

REPO_ROOT="$(pwd -P)"
OFFLINE_ASSET_ROOT="${OFFLINE_ASSET_ROOT:-${REPO_ROOT}/pretrained_models/comparison/offline_assets}"
HF_HOME_SHARED="${HF_HOME_SHARED:-${OFFLINE_ASSET_ROOT}/hf_home}"
HF_HUB_CACHE_SHARED="${HF_HUB_CACHE_SHARED:-${HF_HOME_SHARED}/hub}"
TRANSFORMERS_CACHE_SHARED="${TRANSFORMERS_CACHE_SHARED:-${HF_HOME_SHARED}/transformers}"
SENTENCE_TRANSFORMERS_HOME_SHARED="${SENTENCE_TRANSFORMERS_HOME_SHARED:-${HF_HOME_SHARED}/sentence_transformers}"
TORCH_HOME_SHARED="${TORCH_HOME_SHARED:-${OFFLINE_ASSET_ROOT}/torch_home}"
PANOPTICON_LOCAL_REPO="${PANOPTICON_LOCAL_REPO:-${OFFLINE_ASSET_ROOT}/panopticon_repo}"
CARL_PREFETCH_MARKER="${OFFLINE_ASSET_ROOT}/.carl_prefetched"
PANOPTICON_PREFETCH_MARKER="${OFFLINE_ASSET_ROOT}/.panopticon_prefetched"
SPECAWARE_PREFETCH_MARKER="${OFFLINE_ASSET_ROOT}/.specaware_prefetched"

mkdir -p "${HF_HUB_CACHE_SHARED}" "${TRANSFORMERS_CACHE_SHARED}" "${SENTENCE_TRANSFORMERS_HOME_SHARED}" "${TORCH_HOME_SHARED}"
export HF_HOME="${HF_HOME_SHARED}"
export HF_HUB_CACHE="${HF_HUB_CACHE_SHARED}"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE_SHARED}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE_SHARED}"
export SENTENCE_TRANSFORMERS_HOME="${SENTENCE_TRANSFORMERS_HOME_SHARED}"
export TORCH_HOME="${TORCH_HOME_SHARED}"

# --- Fixed experiment configuration -----------------------------------------
TASK="oxhyperminerals_emit_l2a"
SENSOR="emit"
DECODER="mlp_fusion_head"
IMG_SIZE=224
MAX_EPOCHS="${MAX_EPOCHS:-80}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
BATCH_SIZE="${BATCH_SIZE:-8}"
PROTOCOL="linear"
NUM_CLASSES=3

WANDB_PROJECT="hsi-bench"
DISK_LOG_ROOT="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/niklas/experiments/hsi-bench"

SLURM_EXCLUDE='\"hpdar04c04s02,hpdar04c02s05,hpdar04c01s01,hpdar04c01s02,hpdar04c01s03,hpdar04c03s04,hpdar04c03s05,hpdar04c05s03,hpdar04c05s05,hpdar04c06s02,hpdar04c06s01\"'
SLURM_ACCOUNT="${SLURM_ACCOUNT:-hpda-c}"
SLURM_TIMEOUT_MIN="${SLURM_TIMEOUT_MIN:-}"

BACKBONES=(dofa_b dofav2_b copernicus_fm_b panopticon_b carl specaware_b vit_b vit_b_finetune vit_b_p8 spec_vit_b_p8 spec_vit_b_p16)
BACKBONES_CSV="${BACKBONES_CSV:-}"
if [ -n "$BACKBONES_CSV" ]; then
  IFS=',' read -ra BACKBONES <<< "$BACKBONES_CSV"
fi

DEBUG_MODE="${DEBUG_MODE:-false}"
FORCE_RESUBMIT="${FORCE_RESUBMIT:-false}"

declare -A checkpoint_path_map=(
  ["dofa_b"]="pretrained_models/comparison/dofa_b/DOFA_ViT_base_e100.pth"
  ["dofav2_b"]="pretrained_models/comparison/dofav2_b/dofav2_vit_base_e150.pth"
  ["copernicus_fm_b"]="pretrained_models/comparison/copernicus_fm_b/CopernicusFM_ViT_base_varlang_e100.pth"
  ["panopticon_b"]="pretrained_models/comparison/panopticon/panopticon_vitb14_teacher.pth"
  ["carl"]="pretrained_models/comparison/carl/carl_ssl_checkpoint.ckpt"
  ["specaware_b"]="pretrained_models/comparison/specaware/SpecAware_Base.pth"
)
declare -A weight_init_tag_map=(
  ["dofa_b"]="dofa_pretrained"
  ["dofav2_b"]="dofav2_pretrained"
  ["copernicus_fm_b"]="copernicus_fm_pretrained"
  ["panopticon_b"]="panopticon_pretrained"
  ["carl"]="carl_pretrained"
  ["specaware_b"]="specaware_pretrained"
  ["vit_b"]="random_init"
  ["vit_b_finetune"]="random_init_finetune"
  ["vit_b_p8"]="random_init"
  ["spec_vit_b_p8"]="random_init"
  ["spec_vit_b_p16"]="random_init"
)
declare -A img_size_field_map=(
  ["dofa_b"]="input_size"
  ["dofav2_b"]="input_size"
  ["copernicus_fm_b"]="input_size"
  ["panopticon_b"]="img_size"
  ["carl"]="img_size"
  ["specaware_b"]="img_size"
  ["vit_b"]="img_size"
  ["vit_b_finetune"]="img_size"
  ["vit_b_p8"]="img_size"
  ["spec_vit_b_p8"]="img_size"
  ["spec_vit_b_p16"]="img_size"
)
declare -A patch_size_map=(
  ["dofa_b"]=16
  ["dofav2_b"]=14
  ["copernicus_fm_b"]=16
  ["panopticon_b"]=14
  ["carl"]=8
  ["specaware_b"]=8
  ["vit_b"]=16
  ["vit_b_finetune"]=16
  ["vit_b_p8"]=8
  ["spec_vit_b_p8"]=8
  ["spec_vit_b_p16"]=16
)
declare -A backbone_config_map=(
  ["vit_b_finetune"]="vit_b"
  ["vit_b_p8"]="vit_b"
  ["spec_vit_b_p8"]="spec_vit_b"
  ["spec_vit_b_p16"]="spec_vit_b"
)
declare -A freeze_backbone_map=(
  ["vit_b_finetune"]="false"
)
declare -A protocol_map=(
  ["vit_b_finetune"]="finetune"
)

# EMIT-specific bandwise stats (see header comment) -- shared between the
# DOFA-family and Panopticon-B, same convention as the enmap/desis/eo1
# scripts reusing a single per-sensor stats dir across both families when no
# family-specific version exists.
EMIT_STATS_PATH="data/oxhyperminerals_statistics"

# Per-backbone tap indices for model.tap_indices, matching exactly what
# run_oxhyperminerals_upernet.sh used per backbone (its decoder output_layers
# overrides / defaults) -- see header comment for why this needs to be
# forced rather than left to multi_tap_seg's own auto-default.
declare -A tap_indices_map=(
  ["dofa_b"]="[3,5,7,11]"
  ["dofav2_b"]="[4,6,10,11]"
  ["copernicus_fm_b"]="[3,5,7,11]"
  ["panopticon_b"]="[3,5,7,11]"
  ["carl"]="[1,3,5,7]"
  ["specaware_b"]="[3,5,7,11]"
  # vit_b / vit_b_finetune / vit_b_p8 / spec_vit_b_p8 / spec_vit_b_p16: no
  # entry -- 12-block ViT-B, left unforced (matches the UPerNet script's own
  # convention of not overriding these).
)

# --- Helpers -----------------------------------------------------------------

validate_existing_path() {
  local label="$1" path="$2"
  if [ ! -e "$path" ]; then
    echo "Error: ${label} path does not exist: ${path}" >&2
    return 1
  fi
}

prefetch_timm_model() {
  local model_name="$1"
  echo "Prefetching timm weights for ${model_name}..."
  HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python3 - "$model_name" <<'PYEOF'
import sys
import timm
model = timm.create_model(sys.argv[1], pretrained=True)
del model
PYEOF
}

prefetch_panopticon_repo() {
  if [ -d "${PANOPTICON_LOCAL_REPO}/.git" ]; then
    echo "Using existing Panopticon repo cache at ${PANOPTICON_LOCAL_REPO}"
    return 0
  fi
  echo "Cloning Panopticon repo into ${PANOPTICON_LOCAL_REPO}"
  mkdir -p "$(dirname "${PANOPTICON_LOCAL_REPO}")"
  git clone --depth 1 https://github.com/Panopticon-FM/panopticon.git "${PANOPTICON_LOCAL_REPO}"
}

prefetch_sentence_transformer() {
  local model_name="$1"
  echo "Prefetching sentence-transformer ${model_name}..."
  HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python3 - "$model_name" <<'PYEOF'
import sys
from sentence_transformers import SentenceTransformer
model = SentenceTransformer(sys.argv[1], device="cpu")
del model
PYEOF
}

ensure_offline_assets() {
  local backbone="$1"
  case "$backbone" in
    carl)
      if [ ! -f "${CARL_PREFETCH_MARKER}" ]; then
        prefetch_timm_model "vit_small_patch14_dinov2.lvd142m" || return 1
        prefetch_timm_model "vit_base_patch14_dinov2.lvd142m" || return 1
        prefetch_timm_model "timm/eva02_base_patch14_224.mim_in22k" || return 1
        touch "${CARL_PREFETCH_MARKER}"
      fi
      ;;
    panopticon_b)
      if [ ! -f "${PANOPTICON_PREFETCH_MARKER}" ]; then
        prefetch_panopticon_repo || return 1
        touch "${PANOPTICON_PREFETCH_MARKER}"
      fi
      ;;
    specaware_b)
      if [ ! -f "${SPECAWARE_PREFETCH_MARKER}" ]; then
        prefetch_sentence_transformer "sentence-transformers/all-MiniLM-L6-v2" || return 1
        touch "${SPECAWARE_PREFETCH_MARKER}"
      fi
      ;;
  esac
}

build_weight_overrides() {
  local backbone="$1"
  case "$backbone" in
    vit_b|vit_b_finetune|vit_b_p8|spec_vit_b_p8|spec_vit_b_p16)
      return 0
      ;;
  esac
  local ckpt="${checkpoint_path_map[$backbone]}"
  validate_existing_path "${backbone} checkpoint" "$ckpt" || return 1
  case "$backbone" in
    dofa_b|dofav2_b)
      echo "model.pretrained_weights=${ckpt}"
      ;;
    copernicus_fm_b)
      echo "model.backbone_config.encoder_weights=${ckpt}"
      ;;
    panopticon_b)
      echo "model.backbone_config.checkpoint_path=${ckpt} model.backbone_config.hub_repo=${PANOPTICON_LOCAL_REPO}"
      ;;
    carl)
      echo "model.backbone_config.ssl_ckpt_path=${ckpt}"
      ;;
    specaware_b)
      echo "model.backbone_config.pretrained=${ckpt}"
      ;;
  esac
}

build_norm_overrides() {
  local backbone="$1"
  case "$backbone" in
    dofa_b|dofav2_b|copernicus_fm_b|panopticon_b)
      validate_existing_path "EMIT bandwise stats" "$EMIT_STATS_PATH" || return 1
      echo "data.apply_input_normalization=false data.standardize=true data.standardization_mode=bandwise data.standardization_stats_path=${EMIT_STATS_PATH}"
      ;;
    carl)
      echo "data.apply_input_normalization=false data.standardize=false data.standardization_stats_path=null"
      ;;
    specaware_b|vit_b|vit_b_finetune|vit_b_p8|spec_vit_b_p8|spec_vit_b_p16)
      echo "data.apply_input_normalization=true data.standardize=false data.standardization_stats_path=null"
      ;;
  esac
}

build_tap_overrides() {
  local backbone="$1"
  # multi_tap_is_vit=true: all backbones here are flat ViT-style, never
  # Hiera-like hierarchical.
  local overrides="model.multi_tap_is_vit=true"
  local taps="${tap_indices_map[$backbone]}"
  if [ -n "$taps" ]; then
    overrides="${overrides} model.tap_indices=${taps}"
  fi
  echo "$overrides"
}

enable_offline_runtime_env() {
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
}

has_completed_run() {
  local backbone="$1"
  local experiment_setup="${TASK}_mlpfusion_linear"
  local weight_init="${weight_init_tag_map[$backbone]}"
  local protocol="${protocol_map[$backbone]:-$PROTOCOL}"
  local task_dir="${DISK_LOG_ROOT}/downstream/${experiment_setup}/${SENSOR}/${DECODER}/${backbone}/${protocol}/${weight_init}"

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

submit_backbone() {
  local backbone="$1"
  local experiment_setup="${TASK}_mlpfusion_linear"

  ensure_offline_assets "$backbone" || {
    echo "Warning: failed to prefetch offline assets for ${backbone}. Skipping."
    return 1
  }

  local weight_overrides norm_overrides tap_overrides
  weight_overrides=$(build_weight_overrides "$backbone") || return 1
  norm_overrides=$(build_norm_overrides "$backbone") || return 1
  tap_overrides=$(build_tap_overrides "$backbone")

  local patch_size="${patch_size_map[$backbone]}"
  local img_size_field="${img_size_field_map[$backbone]}"
  local weight_init="${weight_init_tag_map[$backbone]}"
  local protocol="${protocol_map[$backbone]:-$PROTOCOL}"
  local backbone_config="${backbone_config_map[$backbone]:-$backbone}"
  local freeze_backbone="${freeze_backbone_map[$backbone]:-true}"
  local task_name="downstream/${experiment_setup}/${SENSOR}/${DECODER}/${backbone}/${protocol}/${weight_init}"

  local cmd="python src/train.py --config-name=downstream -m"
  cmd+=" experiment=downstream/${TASK}/base"
  cmd+=" data=oxhyperminerals_emit_l2a_10pct"
  cmd+=" backbone@model.backbone_config=${backbone_config}"
  cmd+=" decoder@model.decoder_config=${DECODER}"
  cmd+=" sensor=${SENSOR}"
  cmd+=" sensor@data.sensor_config=${SENSOR}"
  cmd+=" experiment_setup=${experiment_setup}"
  cmd+=" protocol=${protocol}"
  cmd+=" weight_init=${weight_init}"
  cmd+=" backbone_name=${backbone}"
  cmd+=" decoder_name=${DECODER}"
  cmd+=" model.model_type=multi_tap_seg"
  cmd+=" model.num_classes=${NUM_CLASSES}"
  cmd+=" model.freeze_backbone=${freeze_backbone}"
  cmd+=" model.learning_rate=${LEARNING_RATE}"
  cmd+=" model.weight_decay=${WEIGHT_DECAY}"
  cmd+=" model.backbone_config.patch_size=${patch_size}"
  cmd+=" model.backbone_config.${img_size_field}=${IMG_SIZE}"
  cmd+=" data.img_size=${IMG_SIZE}"
  cmd+=" data.batch_size=${BATCH_SIZE}"
  cmd+=" trainer=gpu"
  cmd+=" trainer.max_epochs=${MAX_EPOCHS}"
  cmd+=" paths=default"
  cmd+=" paths.log_dir=${DISK_LOG_ROOT}"
  cmd+=" logger.wandb.project=${WANDB_PROJECT}"
  cmd+=" +logger.wandb.name=${TASK}_${backbone}_mlpfusion"
  cmd+=" task_name=${task_name}"
  cmd+=" hydra/launcher=slurm_one_gpu"
  cmd+=" hydra.launcher.exclude=${SLURM_EXCLUDE}"
  cmd+=" hydra.launcher.account=${SLURM_ACCOUNT}"
  if [ -n "$SLURM_TIMEOUT_MIN" ]; then
    cmd+=" hydra.launcher.timeout_min=${SLURM_TIMEOUT_MIN}"
  fi
  cmd+=" hydra.sweep.subdir=${backbone}"
  cmd+=" ${weight_overrides}"
  cmd+=" ${norm_overrides}"
  cmd+=" ${tap_overrides}"

  echo ""
  echo "===== Backbone: ${backbone} ====="
  echo "$cmd"

  if [ "$DEBUG_MODE" = "true" ]; then
    return 0
  fi

  mkdir -p "${REPO_ROOT}/scripts/downstream/logs"
  local submit_log="${REPO_ROOT}/scripts/downstream/logs/${TASK}_${backbone}_submit.log"
  ( eval "$cmd" > "${submit_log}" 2>&1 ; echo "[${TASK}/${backbone}] submission client exited with code $?" >> "${submit_log}" ) &
  echo "Backgrounded (client log: ${submit_log})"
  sleep 2
}

enable_offline_runtime_env

for backbone in "${BACKBONES[@]}"; do
  if [ "$FORCE_RESUBMIT" != "true" ]; then
    existing_ckpt=$(has_completed_run "$backbone") && {
      echo ""
      echo "===== Backbone: ${backbone} ====="
      echo "Skipping: found an existing checkpoint from a completed run:"
      echo "  ${existing_ckpt}"
      echo "(set FORCE_RESUBMIT=true to resubmit anyway)"
      continue
    }
  fi
  submit_backbone "$backbone" || echo "Warning: ${backbone} run was not submitted (see error above)."
done

echo ""
echo "All jobs submitted."
