#!/bin/bash
# -----------------------------------------------------------------------------
# Wavelength-conditioning ablation: restrict each backbone's input to a named
# spectral subset (rgb / vis / vnir / swir) instead of the full band set, to
# test whether wavelength-conditioned encoders actually exploit the
# wavelength metadata (informative-region subsets should hold performance
# close to full-band; uninformative-region subsets of the same or larger
# band count should degrade much more) rather than treating wavelength as
# decoration.
#
# Only the 5 backbones that support a `wave_list` override and are therefore
# genuinely channel-count-agnostic: dofa_b, dofav2_b, copernicus_fm_b,
# panopticon_b, carl (panopticon_b needed a small extension to
# src/backbones/panopticon.py to add this override -- it previously always
# derived chn_ids/in_chans from the full SensorRegistry metadata with no
# subset hook, unlike the other 4). Fixed-channel backbones (vit_b,
# spec_vit_b, UniverSat) are excluded entirely (per-user decision) since a
# band subset there would mean zero-padding back to the full channel count,
# not a genuine variable-arity input.
#
# Decoder: mlp_fusion_head only. Protocol: linear (frozen backbone) only --
# no finetune variant, no upernet_head, matching what was agreed on.
#
# Groups: rgb (sensor_config.rgb_indices), vis (380-700nm), vnir
# (380-1000nm, the real Si-detector hardware range), swir (1000-2600nm).
# "full" (all bands) is NOT resubmitted here -- it's the existing baseline
# already completed for all 3 tasks x 5 backbones under the plain
# run_oxhyperminerals_mlpfusion.sh / run_pangaea_consistency_mlp_fusion.sh
# runs, reused as the control condition during analysis.
#
# Band-group -> (band_indices, wave_list) resolution is computed ONCE up
# front via scripts/downstream/print_band_group_overrides.py (reads sensor
# YAML/CSVs directly, deliberately not importing src.* -- see that script's
# header for why), cached into band_common_map/wave_nm_map/wave_um_map, and
# reused across all (task, backbone) combos sharing the same (sensor, group)
# pair. Two wave_list unit variants are cached per (sensor, group) since
# backbones disagree on nm vs um for an explicit override -- see
# wave_list_unit_map below.
#
# Tasks: oxhyperminerals_emit_l2a (EMIT, mineral segmentation, SWIR should be
# diagnostic), eurocrops (EnMAP, crop-type segmentation, VNIR/red-edge +
# SWIR moisture should be diagnostic), bnetd (EnMAP, Ivorian forest-type
# segmentation, canopy/species story -- included as the third, non-crop
# domain despite being the slowest task in this batch, a known tradeoff
# accepted explicitly).
#
# Output paths and wandb run names get an extra "/${group}" path segment /
# "_${group}" name suffix beyond the existing full-band convention, so
# results never collide with the full-band baseline or with each other.
# has_completed_run() checks that exact per-group path -- NOT squeue/wandb
# state, same rule as every other script in this benchmark.
#
# Usage:
#   bash scripts/downstream/run_wavelength_ablation.sh
#   TASKS_CSV=oxhyperminerals_emit_l2a BAND_GROUPS_CSV=swir BACKBONES_CSV=dofa_b \
#     DEBUG_MODE=true bash scripts/downstream/run_wavelength_ablation.sh
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

mkdir -p "${HF_HUB_CACHE_SHARED}" "${TRANSFORMERS_CACHE_SHARED}" "${SENTENCE_TRANSFORMERS_HOME_SHARED}" "${TORCH_HOME_SHARED}"
export HF_HOME="${HF_HOME_SHARED}"
export HF_HUB_CACHE="${HF_HUB_CACHE_SHARED}"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE_SHARED}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE_SHARED}"
export SENTENCE_TRANSFORMERS_HOME="${SENTENCE_TRANSFORMERS_HOME_SHARED}"
export TORCH_HOME="${TORCH_HOME_SHARED}"

# --- Fixed experiment configuration -----------------------------------------
TASKS=(oxhyperminerals_emit_l2a eurocrops bnetd)
TASKS_CSV="${TASKS_CSV:-}"
if [ -n "$TASKS_CSV" ]; then
  IFS=',' read -ra TASKS <<< "$TASKS_CSV"
fi

BAND_GROUPS=(rgb vis vnir swir)
BAND_GROUPS_CSV="${BAND_GROUPS_CSV:-}"
if [ -n "$BAND_GROUPS_CSV" ]; then
  IFS=',' read -ra BAND_GROUPS <<< "$BAND_GROUPS_CSV"
fi

BACKBONES=(dofa_b dofav2_b copernicus_fm_b panopticon_b carl)
BACKBONES_CSV="${BACKBONES_CSV:-}"
if [ -n "$BACKBONES_CSV" ]; then
  IFS=',' read -ra BACKBONES <<< "$BACKBONES_CSV"
fi

declare -A task_sensor_map=(
  ["oxhyperminerals_emit_l2a"]="emit"
  ["eurocrops"]="enmap"
  ["bnetd"]="enmap"
)
# Extra per-task overrides beyond the shared ones below -- OxHyp needs the
# 10pct data split (see run_oxhyperminerals_mlpfusion.sh's header for why)
# and an explicit num_classes (its experiment base.yaml doesn't set one from
# model defaults the way eurocrops/bnetd's segmentation model config does).
declare -A task_extra_overrides_map=(
  ["oxhyperminerals_emit_l2a"]="data=oxhyperminerals_emit_l2a_10pct model.num_classes=3"
)

DECODER="mlp_fusion_head"
IMG_SIZE=224
MAX_EPOCHS="${MAX_EPOCHS:-80}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
BATCH_SIZE="${BATCH_SIZE:-8}"
PROTOCOL="linear"

WANDB_PROJECT="hsi-bench"
DISK_LOG_ROOT="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/niklas/experiments/hsi-bench"

SLURM_EXCLUDE='\"hpdar04c04s02,hpdar04c02s05,hpdar04c01s01,hpdar04c01s02,hpdar04c01s03,hpdar04c03s04,hpdar04c03s05,hpdar04c05s03,hpdar04c05s05,hpdar04c06s02,hpdar04c06s01\"'
SLURM_ACCOUNT="${SLURM_ACCOUNT:-hpda-c}"
SLURM_TIMEOUT_MIN="${SLURM_TIMEOUT_MIN:-}"

DEBUG_MODE="${DEBUG_MODE:-false}"
FORCE_RESUBMIT="${FORCE_RESUBMIT:-false}"

declare -A checkpoint_path_map=(
  ["dofa_b"]="pretrained_models/comparison/dofa_b/DOFA_ViT_base_e100.pth"
  ["dofav2_b"]="pretrained_models/comparison/dofav2_b/dofav2_vit_base_e150.pth"
  ["copernicus_fm_b"]="pretrained_models/comparison/copernicus_fm_b/CopernicusFM_ViT_base_varlang_e100.pth"
  ["panopticon_b"]="pretrained_models/comparison/panopticon/panopticon_vitb14_teacher.pth"
  ["carl"]="pretrained_models/comparison/carl/carl_ssl_checkpoint.ckpt"
)
declare -A weight_init_tag_map=(
  ["dofa_b"]="dofa_pretrained"
  ["dofav2_b"]="dofav2_pretrained"
  ["copernicus_fm_b"]="copernicus_fm_pretrained"
  ["panopticon_b"]="panopticon_pretrained"
  ["carl"]="carl_pretrained"
)
declare -A img_size_field_map=(
  ["dofa_b"]="input_size"
  ["dofav2_b"]="input_size"
  ["copernicus_fm_b"]="input_size"
  ["panopticon_b"]="img_size"
  ["carl"]="img_size"
)
declare -A patch_size_map=(
  ["dofa_b"]=16
  ["dofav2_b"]=14
  ["copernicus_fm_b"]=16
  ["panopticon_b"]=14
  ["carl"]=8
)
declare -A tap_indices_map=(
  ["dofa_b"]="[3,5,7,11]"
  ["dofav2_b"]="[4,6,10,11]"
  ["copernicus_fm_b"]="[3,5,7,11]"
  ["panopticon_b"]="[3,5,7,11]"
  ["carl"]="[1,3,5,7]"
)

# Backbones disagree on wave_list units when an explicit override is given
# (used AS-IS, no conversion applied): dofa_b/dofav2_b/carl auto-derive
# micrometers (band_centers_nm * 0.001) when no override is given, so an
# explicit override must also be in micrometers; copernicus_fm_b/panopticon_b
# auto-derive nanometers directly. Caught via an end-to-end test -- passing
# raw nanometers to dofa_b trained without any shape/config error but
# produced NaN train_loss from the very first logged step (see
# print_band_group_overrides.py's header for the full story).
declare -A wave_list_unit_map=(
  ["dofa_b"]="um"
  ["dofav2_b"]="um"
  ["copernicus_fm_b"]="nm"
  ["panopticon_b"]="nm"
  ["carl"]="um"
)

# Per-sensor bandwise normalization stats. EMIT stats are OxHyp-specific
# (computed this session, see run_oxhyperminerals_mlpfusion.sh's header);
# ENMAP stats are the pre-existing dofa/panopticon stats dirs already used by
# run_pangaea_consistency_mlp_fusion.sh. These are FULL-band stats files --
# SingleSensorStandardizer now accepts a band_indices param (added this
# session) that slices mean/std to match the subset at load time, so no
# separate per-group stats file is needed.
declare -A dofa_family_stats_path_map=(
  ["emit"]="data/oxhyperminerals_statistics"
  ["enmap"]="data/dofa_statistics"
)
declare -A panopticon_stats_path_map=(
  ["emit"]="data/oxhyperminerals_statistics"
  ["enmap"]="data/panopticon_statistics"
)

# --- Precompute band-group overrides (once) ----------------------------------
# One torch/lightning import chain total instead of one per (task,backbone,group).
# Two wave_list variants (nm/um) are cached per (sensor,group) -- see
# wave_list_unit_map above for why -- and the correct one is selected per
# backbone in submit_backbone().
echo "Resolving band-group overrides..."
declare -A band_common_map
declare -A wave_nm_map
declare -A wave_um_map
declare -A bandwidth_nm_map
while IFS=$'\t' read -r sensor group band_count common wl_nm wl_um bw_nm; do
  band_common_map["${sensor}_${group}"]="$common"
  wave_nm_map["${sensor}_${group}"]="$wl_nm"
  wave_um_map["${sensor}_${group}"]="$wl_um"
  bandwidth_nm_map["${sensor}_${group}"]="$bw_nm"
done < <(python3 "${REPO_ROOT}/scripts/downstream/print_band_group_overrides.py")

for sensor in emit enmap; do
  for group in "${BAND_GROUPS[@]}"; do
    if [ -z "${band_common_map[${sensor}_${group}]:-}" ]; then
      echo "Error: no band-group override resolved for sensor=${sensor} group=${group}." >&2
      exit 1
    fi
  done
done
echo "Band-group overrides resolved for: ${!band_common_map[@]}"

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
  esac
}

build_weight_overrides() {
  local backbone="$1"
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
  esac
}

build_norm_overrides() {
  local backbone="$1"
  local sensor="$2"
  case "$backbone" in
    dofa_b|dofav2_b|copernicus_fm_b)
      local stats_path="${dofa_family_stats_path_map[$sensor]}"
      validate_existing_path "DOFA-family stats (${sensor})" "$stats_path" || return 1
      echo "data.apply_input_normalization=false data.standardize=true data.standardization_mode=bandwise data.standardization_stats_path=${stats_path}"
      ;;
    panopticon_b)
      local stats_path="${panopticon_stats_path_map[$sensor]}"
      validate_existing_path "Panopticon stats (${sensor})" "$stats_path" || return 1
      echo "data.apply_input_normalization=false data.standardize=true data.standardization_mode=bandwise data.standardization_stats_path=${stats_path}"
      ;;
    carl)
      echo "data.apply_input_normalization=false data.standardize=false data.standardization_stats_path=null"
      ;;
  esac
}

build_tap_overrides() {
  local backbone="$1"
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
  local task="$1"
  local backbone="$2"
  local group="$3"
  local sensor="${task_sensor_map[$task]}"
  local experiment_setup="${task}_mlpfusion_linear"
  local weight_init="${weight_init_tag_map[$backbone]}"
  local task_dir="${DISK_LOG_ROOT}/downstream/${experiment_setup}/${sensor}/${DECODER}/${backbone}/${PROTOCOL}/${weight_init}/${group}"

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
  local task="$1"
  local backbone="$2"
  local group="$3"
  local sensor="${task_sensor_map[$task]}"
  local experiment_setup="${task}_mlpfusion_linear"

  ensure_offline_assets "$backbone" || {
    echo "Warning: failed to prefetch offline assets for ${backbone}. Skipping."
    return 1
  }

  local weight_overrides norm_overrides tap_overrides band_overrides wave_unit wave_overrides bandwidth_overrides
  weight_overrides=$(build_weight_overrides "$backbone") || return 1
  norm_overrides=$(build_norm_overrides "$backbone" "$sensor") || return 1
  tap_overrides=$(build_tap_overrides "$backbone")
  band_overrides="${band_common_map[${sensor}_${group}]}"
  wave_unit="${wave_list_unit_map[$backbone]}"
  if [ "$wave_unit" = "nm" ]; then
    wave_overrides="${wave_nm_map[${sensor}_${group}]}"
  else
    wave_overrides="${wave_um_map[${sensor}_${group}]}"
  fi
  # Only copernicus_fm_b declares a separate bandwidth_list field -- other
  # backbones' configs don't have this key, so passing it would hit the same
  # Hydra struct error +data.band_indices needed a + for.
  bandwidth_overrides=""
  if [ "$backbone" = "copernicus_fm_b" ]; then
    bandwidth_overrides="${bandwidth_nm_map[${sensor}_${group}]}"
  fi

  local patch_size="${patch_size_map[$backbone]}"
  local img_size_field="${img_size_field_map[$backbone]}"
  local weight_init="${weight_init_tag_map[$backbone]}"
  local task_name="downstream/${experiment_setup}/${sensor}/${DECODER}/${backbone}/${PROTOCOL}/${weight_init}/${group}"
  local extra_overrides="${task_extra_overrides_map[$task]:-}"

  local cmd="python src/train.py --config-name=downstream -m"
  cmd+=" experiment=downstream/${task}/base"
  cmd+=" backbone@model.backbone_config=${backbone}"
  cmd+=" decoder@model.decoder_config=${DECODER}"
  cmd+=" sensor=${sensor}"
  cmd+=" sensor@data.sensor_config=${sensor}"
  cmd+=" experiment_setup=${experiment_setup}"
  cmd+=" protocol=${PROTOCOL}"
  cmd+=" weight_init=${weight_init}"
  cmd+=" backbone_name=${backbone}"
  cmd+=" decoder_name=${DECODER}"
  cmd+=" model.model_type=multi_tap_seg"
  cmd+=" model.freeze_backbone=true"
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
  cmd+=" +logger.wandb.name=${task}_${backbone}_mlpfusion_${group}"
  cmd+=" task_name=${task_name}"
  cmd+=" hydra/launcher=slurm_one_gpu"
  cmd+=" hydra.launcher.exclude=${SLURM_EXCLUDE}"
  cmd+=" hydra.launcher.account=${SLURM_ACCOUNT}"
  if [ -n "$SLURM_TIMEOUT_MIN" ]; then
    cmd+=" hydra.launcher.timeout_min=${SLURM_TIMEOUT_MIN}"
  fi
  cmd+=" hydra.sweep.subdir=${backbone}_${group}"
  cmd+=" ${weight_overrides}"
  cmd+=" ${norm_overrides}"
  cmd+=" ${tap_overrides}"
  cmd+=" ${band_overrides}"
  cmd+=" ${wave_overrides}"
  if [ -n "$bandwidth_overrides" ]; then
    cmd+=" ${bandwidth_overrides}"
  fi
  if [ -n "$extra_overrides" ]; then
    cmd+=" ${extra_overrides}"
  fi

  echo ""
  echo "===== Task: ${task} | Group: ${group} | Backbone: ${backbone} ====="
  echo "$cmd"

  if [ "$DEBUG_MODE" = "true" ]; then
    return 0
  fi

  mkdir -p "${REPO_ROOT}/scripts/downstream/logs"
  local submit_log="${REPO_ROOT}/scripts/downstream/logs/${task}_${backbone}_mlpfusion_${group}_submit.log"
  ( eval "$cmd" > "${submit_log}" 2>&1 ; echo "[${task}/${backbone}/${group}] submission client exited with code $?" >> "${submit_log}" ) &
  echo "Backgrounded (client log: ${submit_log})"
  sleep 2
}

enable_offline_runtime_env

for task in "${TASKS[@]}"; do
  for group in "${BAND_GROUPS[@]}"; do
    for backbone in "${BACKBONES[@]}"; do
      if [ "$FORCE_RESUBMIT" != "true" ]; then
        existing_ckpt=$(has_completed_run "$task" "$backbone" "$group") && {
          echo ""
          echo "===== Task: ${task} | Group: ${group} | Backbone: ${backbone} ====="
          echo "Skipping: found an existing checkpoint from a completed run:"
          echo "  ${existing_ckpt}"
          echo "(set FORCE_RESUBMIT=true to resubmit anyway)"
          continue
        }
      fi
      submit_backbone "$task" "$backbone" "$group" || echo "Warning: ${task}/${backbone}/${group} run was not submitted (see error above)."
    done
  done
done

echo ""
echo "All jobs submitted."
