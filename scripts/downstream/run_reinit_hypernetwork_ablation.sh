#!/bin/bash
# -----------------------------------------------------------------------------
# Reinit-hypernetwork ablation: a fundamentally different kind of "break it"
# than run_wrong_wavelength_ablation.sh. That family keeps the pretrained
# hypernetwork (weight_generator) intact and corrupts its INPUT (the
# wavelength metadata). This one keeps the wavelength input CORRECT and
# instead reinitializes the hypernetwork's OWN learnable parameters --
# weight_generator (a small TransformerEncoder + fc_weight/fc_bias +
# weight_tokens/bias_token) and fclayer -- with a fresh standard init
# (xavier_uniform_ for Linear weights, bias=0.01, normal_(std=0.02) for the
# raw token Parameters -- the exact same init scheme each module already uses
# at construction time, see reinit_hypernetwork() in dofa_encoder.py /
# copernicus_fm_encoder.py), AFTER the pretrained checkpoint is loaded so the
# reinit isn't immediately overwritten.
#
# Question this answers: is the *pretrained, learned* per-band weight
# generation itself doing real work, or would an untrained-but-architecturally
# -identical generator (still correctly wavelength-conditioned, since the
# wavelength input is untouched) work just as well given the rest of the
# frozen pretrained transformer blocks stay intact? This is a sharper test of
# "how much value is really in the encoder vs. the decoder" than the
# wrong-wavelength family, since here the encoder's spectral differentiation
# mechanism is architecturally alive and correctly conditioned -- just
# untrained.
#
# Only 3 backbones apply -- DOFA-B, DOFAv2-B, Copernicus-FM-B are the only
# encoders in this benchmark with an actual weight-generating hypernetwork
# (Dynamic_MLP_OFA / _DynamicMLPSpectral). Panopticon-B and CARL use
# different conditioning mechanisms entirely (no equivalent submodule to
# reinit) and are out of scope here -- see the assistant's design discussion
# for the "bypass" follow-up experiment, not yet built.
#
# Single condition, single seed for now (mirrors wrong_jitter_s0/shuffled_s0's
# "start with 1 seed" precedent): reinit_hypernetwork, seed=0
# (REINIT_SEED below) -- passed via `seed=<N>` so the standard-init draw
# (which is NOT deterministic by default in this benchmark's train.py,
# cfg.seed defaults to null / only seeds if truthy) is reproducible and
# documented, not incidental to whatever the login/compute node's ambient RNG
# state happened to be.
#
# No wave_list override at all -- wavelength metadata is exactly correct
# throughout, matching the existing full-band baseline. The only new override
# is `+model.backbone_config.reinit_weight_generator=true` (needs a `+` since
# it's not declared in the backbone YAMLs, same convention as
# +data.band_indices elsewhere in this benchmark).
#
# Same 5-backbone-ablation scaffolding (env setup, offline-asset prefetch,
# weight/norm/tap override builders, has_completed_run/submit_backbone
# pattern) as run_wrong_wavelength_ablation.sh, trimmed to the 3 applicable
# backbones and with the wave_list precompute step removed entirely.
#
# Output paths / wandb names get a "/reinit_hypernetwork" segment, parallel
# to run_wrong_wavelength_ablation.sh's "/${condition}".
# has_completed_run() checks that exact path -- NOT squeue/wandb state, same
# rule as every other script in this benchmark.
#
# Usage:
#   bash scripts/downstream/run_reinit_hypernetwork_ablation.sh
#   TASKS_CSV=oxhyperminerals_emit_l2a BACKBONES_CSV=dofa_b \
#     DEBUG_MODE=true bash scripts/downstream/run_reinit_hypernetwork_ablation.sh
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

CONDITION="reinit_hypernetwork"
REINIT_SEED="${REINIT_SEED:-0}"

BACKBONES=(dofa_b dofav2_b copernicus_fm_b)
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
)
declare -A weight_init_tag_map=(
  ["dofa_b"]="dofa_pretrained"
  ["dofav2_b"]="dofav2_pretrained"
  ["copernicus_fm_b"]="copernicus_fm_pretrained"
)
declare -A img_size_field_map=(
  ["dofa_b"]="input_size"
  ["dofav2_b"]="input_size"
  ["copernicus_fm_b"]="input_size"
)
declare -A patch_size_map=(
  ["dofa_b"]=16
  ["dofav2_b"]=14
  ["copernicus_fm_b"]=16
)
declare -A tap_indices_map=(
  ["dofa_b"]="[3,5,7,11]"
  ["dofav2_b"]="[4,6,10,11]"
  ["copernicus_fm_b"]="[3,5,7,11]"
)
declare -A wave_list_unit_map=(
  ["dofa_b"]="um"
  ["dofav2_b"]="um"
  ["copernicus_fm_b"]="nm"
)
declare -A dofa_family_stats_path_map=(
  ["emit"]="data/oxhyperminerals_statistics"
  ["enmap"]="data/dofa_statistics"
)

# --- Helpers -----------------------------------------------------------------

validate_existing_path() {
  local label="$1" path="$2"
  if [ ! -e "$path" ]; then
    echo "Error: ${label} path does not exist: ${path}" >&2
    return 1
  fi
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
  esac
}

build_norm_overrides() {
  local backbone="$1"
  local sensor="$2"
  local stats_path="${dofa_family_stats_path_map[$sensor]}"
  validate_existing_path "DOFA-family stats (${sensor})" "$stats_path" || return 1
  echo "data.apply_input_normalization=false data.standardize=true data.standardization_mode=bandwise data.standardization_stats_path=${stats_path}"
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
  local sensor="${task_sensor_map[$task]}"
  local experiment_setup="${task}_mlpfusion_linear"
  local weight_init="${weight_init_tag_map[$backbone]}"
  local task_dir="${DISK_LOG_ROOT}/downstream/${experiment_setup}/${sensor}/${DECODER}/${backbone}/${PROTOCOL}/${weight_init}/${CONDITION}"

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
  local sensor="${task_sensor_map[$task]}"
  local experiment_setup="${task}_mlpfusion_linear"

  local weight_overrides norm_overrides tap_overrides
  weight_overrides=$(build_weight_overrides "$backbone") || return 1
  norm_overrides=$(build_norm_overrides "$backbone" "$sensor") || return 1
  tap_overrides=$(build_tap_overrides "$backbone")

  local patch_size="${patch_size_map[$backbone]}"
  local img_size_field="${img_size_field_map[$backbone]}"
  local weight_init="${weight_init_tag_map[$backbone]}"
  local task_name="downstream/${experiment_setup}/${sensor}/${DECODER}/${backbone}/${PROTOCOL}/${weight_init}/${CONDITION}"
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
  cmd+=" +logger.wandb.name=${task}_${backbone}_mlpfusion_${CONDITION}"
  cmd+=" task_name=${task_name}"
  cmd+=" seed=${REINIT_SEED}"
  cmd+=" hydra/launcher=slurm_one_gpu"
  cmd+=" hydra.launcher.exclude=${SLURM_EXCLUDE}"
  cmd+=" hydra.launcher.account=${SLURM_ACCOUNT}"
  if [ -n "$SLURM_TIMEOUT_MIN" ]; then
    cmd+=" hydra.launcher.timeout_min=${SLURM_TIMEOUT_MIN}"
  fi
  cmd+=" hydra.sweep.subdir=${backbone}_${CONDITION}"
  cmd+=" ${weight_overrides}"
  cmd+=" ${norm_overrides}"
  cmd+=" ${tap_overrides}"
  cmd+=" +model.backbone_config.reinit_weight_generator=true"
  if [ -n "$extra_overrides" ]; then
    cmd+=" ${extra_overrides}"
  fi

  echo ""
  echo "===== Task: ${task} | Backbone: ${backbone} | Condition: ${CONDITION} ====="
  echo "$cmd"

  if [ "$DEBUG_MODE" = "true" ]; then
    return 0
  fi

  mkdir -p "${REPO_ROOT}/scripts/downstream/logs"
  local submit_log="${REPO_ROOT}/scripts/downstream/logs/${task}_${backbone}_mlpfusion_${CONDITION}_submit.log"
  ( eval "$cmd" > "${submit_log}" 2>&1 ; echo "[${task}/${backbone}/${CONDITION}] submission client exited with code $?" >> "${submit_log}" ) &
  echo "Backgrounded (client log: ${submit_log})"
  sleep 2
}

enable_offline_runtime_env

for task in "${TASKS[@]}"; do
  for backbone in "${BACKBONES[@]}"; do
    if [ "$FORCE_RESUBMIT" != "true" ]; then
      existing_ckpt=$(has_completed_run "$task" "$backbone") && {
        echo ""
        echo "===== Task: ${task} | Backbone: ${backbone} | Condition: ${CONDITION} ====="
        echo "Skipping: found an existing checkpoint from a completed run:"
        echo "  ${existing_ckpt}"
        echo "(set FORCE_RESUBMIT=true to resubmit anyway)"
        continue
      }
    fi
    submit_backbone "$task" "$backbone" || echo "Warning: ${task}/${backbone}/${CONDITION} run was not submitted (see error above)."
  done
done

echo ""
echo "All jobs submitted."
