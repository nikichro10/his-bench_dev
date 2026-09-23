#!/bin/bash
# -----------------------------------------------------------------------------
# Wrong-wavelength ablation: keep every band (full spectrum, same input as the
# existing full-band baseline) but feed each wavelength-conditioned backbone
# DELIBERATELY WRONG center-wavelength metadata for the real pixel data. Not
# the same question as run_wavelength_ablation.sh/run_random_band_ablation.sh
# (which restrict WHICH/HOW MANY bands the model sees, wave_list always
# correct for the bands actually given) -- here band count and content never
# change, only whether the model's belief about "what wavelength is this
# channel" is right. Directly analogous to the user's earlier multispectral
# pangaea-bench experiment feeding wrong central wavelengths, to test whether
# a backbone's wavelength conditioning is actually load-bearing or just a
# fixed per-channel embedding it could equally have learned positionally.
#
# Five conditions (see print_wrong_wavelength_overrides.py's header for the
# exact construction):
#   - wrong_permuted_s0/s1/s2: the true wavelength VALUES, randomly
#     reassigned across bands (3 seeds, so a headline effect isn't resting on
#     one lucky/unlucky shuffle). Pixel DATA stays in original band order --
#     only the label is wrong.
#   - wrong_constant: every band told the sensor's median true wavelength --
#     removes all cross-channel differentiation from the signal entirely.
#   - wrong_jitter_s0: a milder, more realistic corruption -- each band keeps
#     its OWN true wavelength, nudged by an independently-drawn perturbation
#     (magnitude uniform in [20,50]nm, random sign per band). Tests
#     calibration-drift-sized noise rather than gross misidentification. Only
#     1 seed for now (per explicit instruction), unlike wrong_permuted's 3.
#   - wrong_jitter_50_100_s0: same construction as wrong_jitter_s0, but with
#     the per-band offset magnitude doubled to [50,100]nm -- tests whether
#     the near-total tolerance of the smaller jitter range degrades once the
#     noise gets larger. Only 1 seed for now, same as wrong_jitter_s0.
#   - shuffled_s0: a different question from the three above -- reuses
#     wrong_permuted_s0's exact permutation, but now shuffles the pixel DATA
#     by that same permutation too (via a `+data.band_indices=<perm>`
#     override, unlike the other three conditions), so the (data, wavelength)
#     pairing at every position stays CORRECT, just reordered. Tests whether
#     the backbone needs canonical band order even when told the truth at
#     every position. Only 1 seed for now.
# "full" (correct wave_list) is NOT resubmitted here -- it's the existing
# baseline, reused as the control condition during analysis.
#
# Band WIDTHS are the TRUE, unpermuted widths (in original band order) --
# never corrupted by either condition, so this stays a single-factor test of
# center-wavelength correctness. Passed explicitly ONLY for copernicus_fm_b
# (same convention as run_wavelength_ablation.sh/run_random_band_ablation.sh):
# CopernicusFMBasePatch16 only auto-derives wv_list/bw_list TOGETHER from
# SensorRegistry when wave_list is None -- since wave_list is always
# overridden here, bandwidth_list must be given too, or
# `self.bw_list = self.bandwidth_list[self.sensor]` crashes on `None[...]`.
# Caught via an end-to-end test before mass submission (see
# print_wrong_wavelength_overrides.py's header for the full story).
#
# Same 5 backbones/decoder/protocol as run_wavelength_ablation.sh: dofa_b,
# dofav2_b, copernicus_fm_b, panopticon_b, carl -- mlp_fusion_head, frozen
# linear probe. Tasks: oxhyperminerals_emit_l2a/eurocrops/bnetd, plus
# bdforet (added later, same EnMAP sensor as eurocrops/bnetd -- see
# run_small_band_ablation.sh's task_sensor_map for the established
# convention; no task_extra_overrides_map entry needed, same as
# eurocrops/bnetd). UniverSat is handled by the separate
# run_wrong_wavelength_universat.sh script, same split rationale as every
# other ablation script in this benchmark.
#
# Output paths / wandb names get an extra "/${condition}" segment (e.g.
# "wrong_constant"), parallel to run_wavelength_ablation.sh's "/${group}".
# has_completed_run() checks that exact path -- NOT squeue/wandb state, same
# rule as every other script in this benchmark.
#
# NOT SUBMITTED YET as of writing -- written ahead of time while the
# random-band-ablation batch (n20 size + vit_b baseline, ~107 jobs) is still
# running, per explicit request to wait before submitting this one.
#
# Usage:
#   bash scripts/downstream/run_wrong_wavelength_ablation.sh
#   TASKS_CSV=oxhyperminerals_emit_l2a CONDITIONS_CSV=wrong_constant BACKBONES_CSV=dofa_b \
#     DEBUG_MODE=true bash scripts/downstream/run_wrong_wavelength_ablation.sh
# -----------------------------------------------------------------------------
set -e

eval "$(micromamba shell hook --shell bash)"
micromamba activate hsi-bench

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
# See run_random_band_ablation.sh's header comment for why: compute nodes
# currently can't reach the internet post-maintenance, so wandb.init() dies
# before training starts. Offline mode still produces the local
# "Best ckpt path:" log line has_completed_run()/the notebook read. Sync
# later with `wandb sync <run dir>` once connectivity is confirmed back.
export WANDB_MODE=offline

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
TASKS=(oxhyperminerals_emit_l2a eurocrops bnetd bdforet)
TASKS_CSV="${TASKS_CSV:-}"
if [ -n "$TASKS_CSV" ]; then
  IFS=',' read -ra TASKS <<< "$TASKS_CSV"
fi

CONDITIONS=(wrong_permuted_s0 wrong_permuted_s1 wrong_permuted_s2 wrong_constant wrong_jitter_s0 wrong_jitter_50_100_s0 shuffled_s0)
CONDITIONS_CSV="${CONDITIONS_CSV:-}"
if [ -n "$CONDITIONS_CSV" ]; then
  IFS=',' read -ra CONDITIONS <<< "$CONDITIONS_CSV"
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
  ["bdforet"]="enmap"
)
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
declare -A wave_list_unit_map=(
  ["dofa_b"]="um"
  ["dofav2_b"]="um"
  ["copernicus_fm_b"]="nm"
  ["panopticon_b"]="nm"
  ["carl"]="um"
)
declare -A dofa_family_stats_path_map=(
  ["emit"]="data/oxhyperminerals_statistics"
  ["enmap"]="data/dofa_statistics"
)
declare -A panopticon_stats_path_map=(
  ["emit"]="data/oxhyperminerals_statistics"
  ["enmap"]="data/panopticon_statistics"
)

# --- Precompute wrong-wavelength overrides (once) ----------------------------
echo "Resolving wrong-wavelength overrides..."
declare -A wave_nm_map
declare -A wave_um_map
declare -A bandwidth_nm_map
declare -A band_indices_map
while IFS=$'\t' read -r sensor condition wl_nm wl_um bw_nm band_idx; do
  wave_nm_map["${sensor}_${condition}"]="$wl_nm"
  wave_um_map["${sensor}_${condition}"]="$wl_um"
  bandwidth_nm_map["${sensor}_${condition}"]="$bw_nm"
  band_indices_map["${sensor}_${condition}"]="$band_idx"
done < <(python3 "${REPO_ROOT}/scripts/downstream/print_wrong_wavelength_overrides.py")

for sensor in emit enmap; do
  for condition in "${CONDITIONS[@]}"; do
    if [ -z "${wave_nm_map[${sensor}_${condition}]:-}" ]; then
      echo "Error: no wrong-wavelength override resolved for sensor=${sensor} condition=${condition}." >&2
      exit 1
    fi
  done
done
echo "Wrong-wavelength overrides resolved for: ${!wave_nm_map[@]}"

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
  local condition="$3"
  local sensor="${task_sensor_map[$task]}"
  local experiment_setup="${task}_mlpfusion_linear"
  local weight_init="${weight_init_tag_map[$backbone]}"
  local task_dir="${DISK_LOG_ROOT}/downstream/${experiment_setup}/${sensor}/${DECODER}/${backbone}/${PROTOCOL}/${weight_init}/${condition}"

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
  local condition="$3"
  local sensor="${task_sensor_map[$task]}"
  local experiment_setup="${task}_mlpfusion_linear"

  ensure_offline_assets "$backbone" || {
    echo "Warning: failed to prefetch offline assets for ${backbone}. Skipping."
    return 1
  }

  local weight_overrides norm_overrides tap_overrides wave_unit wave_overrides bandwidth_overrides band_indices_overrides
  weight_overrides=$(build_weight_overrides "$backbone") || return 1
  norm_overrides=$(build_norm_overrides "$backbone" "$sensor") || return 1
  tap_overrides=$(build_tap_overrides "$backbone")
  wave_unit="${wave_list_unit_map[$backbone]}"
  if [ "$wave_unit" = "nm" ]; then
    wave_overrides="${wave_nm_map[${sensor}_${condition}]}"
  else
    wave_overrides="${wave_um_map[${sensor}_${condition}]}"
  fi
  # Only copernicus_fm_b declares a separate bandwidth_list field -- other
  # backbones' configs don't have this key, so passing it would hit the same
  # "not in struct" error +data.band_indices needed a + for.
  bandwidth_overrides=""
  if [ "$backbone" = "copernicus_fm_b" ]; then
    bandwidth_overrides="${bandwidth_nm_map[${sensor}_${condition}]}"
  fi
  # Only shuffled_s0 has a non-empty band_indices entry -- the other
  # conditions never reorder pixel data, only the wavelength label.
  band_indices_overrides="${band_indices_map[${sensor}_${condition}]}"

  local patch_size="${patch_size_map[$backbone]}"
  local img_size_field="${img_size_field_map[$backbone]}"
  local weight_init="${weight_init_tag_map[$backbone]}"
  local task_name="downstream/${experiment_setup}/${sensor}/${DECODER}/${backbone}/${PROTOCOL}/${weight_init}/${condition}"
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
  cmd+=" +logger.wandb.name=${task}_${backbone}_mlpfusion_${condition}"
  cmd+=" task_name=${task_name}"
  cmd+=" hydra/launcher=slurm_one_gpu"
  cmd+=" hydra.launcher.exclude=${SLURM_EXCLUDE}"
  cmd+=" hydra.launcher.account=${SLURM_ACCOUNT}"
  if [ -n "$SLURM_TIMEOUT_MIN" ]; then
    cmd+=" hydra.launcher.timeout_min=${SLURM_TIMEOUT_MIN}"
  fi
  cmd+=" hydra.sweep.subdir=${backbone}_${condition}"
  cmd+=" ${weight_overrides}"
  cmd+=" ${norm_overrides}"
  cmd+=" ${tap_overrides}"
  cmd+=" ${wave_overrides}"
  if [ -n "$bandwidth_overrides" ]; then
    cmd+=" ${bandwidth_overrides}"
  fi
  if [ -n "$band_indices_overrides" ]; then
    cmd+=" ${band_indices_overrides}"
  fi
  if [ -n "$extra_overrides" ]; then
    cmd+=" ${extra_overrides}"
  fi

  echo ""
  echo "===== Task: ${task} | Condition: ${condition} | Backbone: ${backbone} ====="
  echo "$cmd"

  if [ "$DEBUG_MODE" = "true" ]; then
    return 0
  fi

  mkdir -p "${REPO_ROOT}/scripts/downstream/logs"
  local submit_log="${REPO_ROOT}/scripts/downstream/logs/${task}_${backbone}_mlpfusion_${condition}_submit.log"
  ( eval "$cmd" > "${submit_log}" 2>&1 ; echo "[${task}/${backbone}/${condition}] submission client exited with code $?" >> "${submit_log}" ) &
  echo "Backgrounded (client log: ${submit_log})"
  sleep 2
}

enable_offline_runtime_env

for task in "${TASKS[@]}"; do
  for condition in "${CONDITIONS[@]}"; do
    for backbone in "${BACKBONES[@]}"; do
      if [ "$FORCE_RESUBMIT" != "true" ]; then
        existing_ckpt=$(has_completed_run "$task" "$backbone" "$condition") && {
          echo ""
          echo "===== Task: ${task} | Condition: ${condition} | Backbone: ${backbone} ====="
          echo "Skipping: found an existing checkpoint from a completed run:"
          echo "  ${existing_ckpt}"
          echo "(set FORCE_RESUBMIT=true to resubmit anyway)"
          continue
        }
      fi
      submit_backbone "$task" "$backbone" "$condition" || echo "Warning: ${task}/${backbone}/${condition} run was not submitted (see error above)."
    done
  done
done

echo ""
echo "All jobs submitted."
