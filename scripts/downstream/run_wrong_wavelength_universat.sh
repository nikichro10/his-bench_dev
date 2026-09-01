#!/bin/bash
# -----------------------------------------------------------------------------
# Wrong-wavelength ablation for UniverSat-B -- the UniverSat counterpart to
# run_wrong_wavelength_ablation.sh (which covers the 5 mlp_fusion_head-capable
# encoders), mirroring how run_random_band_universat.sh already split
# UniverSat out from run_random_band_ablation.sh (different decoder/
# model_type/img_size -- see that script's header for the full rationale).
#
# Same two conditions as run_wrong_wavelength_ablation.sh (see
# print_wrong_wavelength_overrides.py's header for the exact construction):
# wrong_permuted_s0/s1/s2 (true wavelength values, randomly reassigned across
# bands) and wrong_constant (every band told the sensor's median true
# wavelength). Full band count throughout -- no data.band_indices/
# probe_in_chans override (UniverSat's ConvSegmentor doesn't need the latter
# anyway, see run_random_band_universat.sh's header).
#
# wave_list override reuses the same um-only mechanism
# run_random_band_universat.sh already validated for UniverSat
# (src/backbones/universat_encoder.py's wave_list param, added this session).
#
# img_size stays per-sensor NATIVE (64 for EMIT, 128 for EnMAP), same
# deliberate choice as every other UniverSat script in this benchmark.
#
# Only 1 "backbone" (UniverSat itself), so this is TASKS x CONDITIONS = 12
# runs total, not TASKS x CONDITIONS x BACKBONES.
#
# NOT SUBMITTED YET as of writing -- written ahead of time while the
# random-band-ablation batch (n20 size + vit_b baseline, ~107 jobs) is still
# running, per explicit request to wait before submitting this one.
#
# Usage:
#   bash scripts/downstream/run_wrong_wavelength_universat.sh
#   TASKS_CSV=oxhyperminerals_emit_l2a CONDITIONS_CSV=wrong_constant \
#     DEBUG_MODE=true bash scripts/downstream/run_wrong_wavelength_universat.sh
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
TASKS=(oxhyperminerals_emit_l2a eurocrops bnetd)
TASKS_CSV="${TASKS_CSV:-}"
if [ -n "$TASKS_CSV" ]; then
  IFS=',' read -ra TASKS <<< "$TASKS_CSV"
fi

CONDITIONS=(wrong_permuted_s0 wrong_permuted_s1 wrong_permuted_s2 wrong_constant)
CONDITIONS_CSV="${CONDITIONS_CSV:-}"
if [ -n "$CONDITIONS_CSV" ]; then
  IFS=',' read -ra CONDITIONS <<< "$CONDITIONS_CSV"
fi

declare -A task_sensor_map=(
  ["oxhyperminerals_emit_l2a"]="emit"
  ["eurocrops"]="enmap"
  ["bnetd"]="enmap"
)
declare -A task_img_size_map=(
  ["oxhyperminerals_emit_l2a"]=64
  ["eurocrops"]=128
  ["bnetd"]=128
)
declare -A task_extra_overrides_map=(
  ["oxhyperminerals_emit_l2a"]="data=oxhyperminerals_emit_l2a_10pct model.num_classes=3"
)

BACKBONE="universat_b"
DECODER="universat_conv_head"
PATCH_SIZE=1
LATENT_PATCH_SIZE=2
MAX_EPOCHS="${MAX_EPOCHS:-80}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
BATCH_SIZE="${BATCH_SIZE:-4}"
PROTOCOL="linear"
WEIGHT_INIT="universat_pretrained"

WANDB_PROJECT="hsi-bench"
DISK_LOG_ROOT="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/niklas/experiments/hsi-bench"

SLURM_EXCLUDE='\"hpdar04c04s02,hpdar04c02s05,hpdar04c01s01,hpdar04c01s02,hpdar04c01s03,hpdar04c03s04,hpdar04c03s05,hpdar04c05s03,hpdar04c05s05,hpdar04c06s02,hpdar04c06s01\"'
SLURM_ACCOUNT="${SLURM_ACCOUNT:-hpda-c}"
# Every condition in THIS script runs UniverSat at its full, unrestricted
# band count (202 EnMAP / 244 EMIT) -- unlike every other UniverSat script in
# this benchmark, which only ever ran it band-restricted (n20/rgb/vis/vnir/
# swir subsets, all much cheaper). The launcher's default timeout_min=600
# (10h) wasn't tuned for this case: the wrong_permuted_s0/bnetd test job hit
# it repeatedly and was eventually abandoned by submitit
# ("timed-out too many times") after reaching only 33/79 epochs in ~9h at a
# steady 0.37 it/s -- extrapolated, 80 epochs needs ~22h. 1440min (24h) here
# gives headroom above that observed rate. Partition hpda2_compute_gpu allows
# up to 10 days walltime (`sinfo -p hpda2_compute_gpu`), so this is well
# within bounds.
SLURM_TIMEOUT_MIN="${SLURM_TIMEOUT_MIN:-1440}"

DEBUG_MODE="${DEBUG_MODE:-false}"
FORCE_RESUBMIT="${FORCE_RESUBMIT:-false}"

declare -A dofa_family_stats_path_map=(
  ["emit"]="data/oxhyperminerals_statistics"
  ["enmap"]="data/dofa_statistics"
)

# --- Precompute wrong-wavelength overrides (once) ----------------------------
echo "Resolving wrong-wavelength overrides..."
declare -A wave_um_map
while IFS=$'\t' read -r sensor condition wl_nm wl_um bw_nm; do
  wave_um_map["${sensor}_${condition}"]="$wl_um"
done < <(python3 "${REPO_ROOT}/scripts/downstream/print_wrong_wavelength_overrides.py")

for sensor in emit enmap; do
  for condition in "${CONDITIONS[@]}"; do
    if [ -z "${wave_um_map[${sensor}_${condition}]:-}" ]; then
      echo "Error: no wrong-wavelength override resolved for sensor=${sensor} condition=${condition}." >&2
      exit 1
    fi
  done
done
echo "Wrong-wavelength overrides resolved for: ${!wave_um_map[@]}"

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

build_norm_overrides() {
  local sensor="$1"
  local stats_path="${dofa_family_stats_path_map[$sensor]}"
  validate_existing_path "bandwise stats (${sensor})" "$stats_path" || return 1
  echo "data.apply_input_normalization=false data.standardize=true data.standardization_mode=bandwise data.standardization_stats_path=${stats_path}"
}

enable_offline_runtime_env() {
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
}

has_completed_run() {
  local task="$1"
  local condition="$2"
  local sensor="${task_sensor_map[$task]}"
  local experiment_setup="${task}_convseg_linear"
  local task_dir="${DISK_LOG_ROOT}/downstream/${experiment_setup}/${sensor}/${DECODER}/${BACKBONE}/${PROTOCOL}/${WEIGHT_INIT}/${condition}"

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
  local task="$1"
  local condition="$2"
  local sensor="${task_sensor_map[$task]}"
  local img_size="${task_img_size_map[$task]}"
  local experiment_setup="${task}_convseg_linear"

  ensure_offline_assets || {
    echo "Warning: failed to prefetch offline assets for UniverSat. Skipping."
    return 1
  }

  local norm_overrides wave_overrides
  norm_overrides=$(build_norm_overrides "$sensor") || return 1
  wave_overrides="${wave_um_map[${sensor}_${condition}]}"

  local task_name="downstream/${experiment_setup}/${sensor}/${DECODER}/${BACKBONE}/${PROTOCOL}/${WEIGHT_INIT}/${condition}"
  local extra_overrides="${task_extra_overrides_map[$task]:-}"

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
  cmd+=" model.backbone_config.img_size=${img_size}"
  cmd+=" model.backbone_config.pretrained=true"
  cmd+=" data.img_size=${img_size}"
  cmd+=" data.batch_size=${BATCH_SIZE}"
  cmd+=" trainer=gpu"
  cmd+=" trainer.max_epochs=${MAX_EPOCHS}"
  cmd+=" paths=default"
  cmd+=" paths.log_dir=${DISK_LOG_ROOT}"
  cmd+=" logger.wandb.project=${WANDB_PROJECT}"
  cmd+=" +logger.wandb.name=${task}_${BACKBONE}_${condition}"
  cmd+=" task_name=${task_name}"
  cmd+=" hydra/launcher=slurm_one_gpu"
  cmd+=" hydra.launcher.exclude=${SLURM_EXCLUDE}"
  cmd+=" hydra.launcher.account=${SLURM_ACCOUNT}"
  if [ -n "$SLURM_TIMEOUT_MIN" ]; then
    cmd+=" hydra.launcher.timeout_min=${SLURM_TIMEOUT_MIN}"
  fi
  cmd+=" hydra.sweep.subdir=${BACKBONE}_${condition}"
  cmd+=" ${norm_overrides}"
  cmd+=" ${wave_overrides}"
  if [ -n "$extra_overrides" ]; then
    cmd+=" ${extra_overrides}"
  fi

  echo ""
  echo "===== Task: ${task} | Condition: ${condition} ====="
  echo "$cmd"

  if [ "$DEBUG_MODE" = "true" ]; then
    return 0
  fi

  mkdir -p "${REPO_ROOT}/scripts/downstream/logs"
  local submit_log="${REPO_ROOT}/scripts/downstream/logs/${task}_${BACKBONE}_${condition}_submit.log"
  ( eval "$cmd" > "${submit_log}" 2>&1 ; echo "[${task}/${condition}] submission client exited with code $?" >> "${submit_log}" ) &
  echo "Backgrounded (client log: ${submit_log})"
  sleep 2
}

enable_offline_runtime_env

for task in "${TASKS[@]}"; do
  for condition in "${CONDITIONS[@]}"; do
    if [ "$FORCE_RESUBMIT" != "true" ]; then
      existing_ckpt=$(has_completed_run "$task" "$condition") && {
        echo ""
        echo "===== Task: ${task} | Condition: ${condition} ====="
        echo "Skipping: found an existing checkpoint from a completed run:"
        echo "  ${existing_ckpt}"
        echo "(set FORCE_RESUBMIT=true to resubmit anyway)"
        continue
      }
    fi
    submit_job "$task" "$condition" || echo "Warning: ${task}/${condition} run was not submitted (see error above)."
  done
done

echo ""
echo "All jobs submitted."
