#!/bin/bash
# -----------------------------------------------------------------------------
# Band-restriction ablation for vit_b -- a random-init, non-wavelength-
# conditioned ViT-B/16 baseline, run across the SAME semantic-group and
# random-subset band conditions as run_wavelength_ablation.sh /
# run_random_band_ablation.sh. Tests whether a model with no notion of
# wavelength at all (plain timm ViT, in_chans set once at construction,
# unlike dofa_b/dofav2_b/copernicus_fm_b/panopticon_b/carl which all take a
# wave_list override) behaves any differently under band restriction --
# e.g. whether the "random beats semantic" pattern seen for the
# wavelength-conditioned encoders also holds for a backbone that can't tell
# WHICH bands it got, only how many.
#
# vit_b's full-band (all-bands) mlp_fusion_head/linear baseline already
# exists on disk for all 3 tasks below (run via
# run_pangaea_consistency_mlp_fusion.sh / run_oxhyperminerals_mlpfusion.sh)
# -- NOT resubmitted here, only the band-restricted conditions.
#
# Conditions:
#   - 4 semantic groups (rgb/vis/vnir/swir), overrides from
#     print_band_group_overrides.py -- same groups run_wavelength_ablation.sh
#     uses for the other 5 encoders.
#   - 5 random sizes (rgb/n20/vis/vnir/swir band counts) x 3 seeds, overrides
#     from print_random_band_overrides.py -- same draws
#     run_random_band_ablation.sh/run_random_band_universat.sh use for their
#     encoders (n20 is a fixed 20-band draw with no semantic counterpart --
#     see that script's header).
# 19 conditions x 3 tasks = 57 runs total.
#
# vit_b has NO wave_list/bandwidth_list field at all (configs/backbone/
# vit_b.yaml is a plain timm wrapper -- src/backbones/timm_wrapper.py just
# passes in_chans straight to timm.create_model) -- passing either override
# would hit Hydra's "not in struct" error. The only thing vit_b needs to
# become band-count-agnostic is model.backbone_config.in_chans=<N>, a plain
# override of an EXISTING field (no leading '+'), unlike the wave_list
# encoders' probe_in_chans-only requirement. Still also passes
# model.probe_in_chans=<N> (emitted by both override scripts' "common"
# field) since mlp_fusion_head's eager multi-tap shape probe needs it
# regardless of backbone -- see print_band_group_overrides.py's header.
#
# patch_size=16, protocol=linear (frozen), weight_init=random_init -- exactly
# matching how run_pangaea_consistency_mlp_fusion.sh/
# run_oxhyperminerals_mlpfusion.sh already run vit_b's full-band baseline, so
# these results are directly comparable to that baseline. No checkpoint to
# load (random init) and no per-sensor normalization stats needed (uses the
# datamodule's generic sensor-aware normalization, not bandwise stats keyed
# to a specific pretrained encoder's stats file).
#
# Already-completed (task, condition) combinations are skipped automatically
# via the "Best ckpt path:" log line -- same rule as every other script in
# this benchmark, never squeue/wandb state alone.
#
# Usage:
#   bash scripts/downstream/run_vit_b_band_ablation.sh
#   TASKS_CSV=oxhyperminerals_emit_l2a CONDITIONS_CSV=rand_n20_s0 \
#     DEBUG_MODE=true bash scripts/downstream/run_vit_b_band_ablation.sh
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

# --- Fixed experiment configuration -----------------------------------------
TASKS=(oxhyperminerals_emit_l2a eurocrops bnetd)
TASKS_CSV="${TASKS_CSV:-}"
if [ -n "$TASKS_CSV" ]; then
  IFS=',' read -ra TASKS <<< "$TASKS_CSV"
fi

CONDITIONS=(rgb vis vnir swir rand_rgb_s0 rand_rgb_s1 rand_rgb_s2 rand_n20_s0 rand_n20_s1 rand_n20_s2 rand_vis_s0 rand_vis_s1 rand_vis_s2 rand_vnir_s0 rand_vnir_s1 rand_vnir_s2 rand_swir_s0 rand_swir_s1 rand_swir_s2)
CONDITIONS_CSV="${CONDITIONS_CSV:-}"
if [ -n "$CONDITIONS_CSV" ]; then
  IFS=',' read -ra CONDITIONS <<< "$CONDITIONS_CSV"
fi

declare -A task_sensor_map=(
  ["oxhyperminerals_emit_l2a"]="emit"
  ["eurocrops"]="enmap"
  ["bnetd"]="enmap"
)
declare -A task_extra_overrides_map=(
  ["oxhyperminerals_emit_l2a"]="data=oxhyperminerals_emit_l2a_10pct model.num_classes=3"
)

BACKBONE="vit_b"
DECODER="mlp_fusion_head"
IMG_SIZE=224
PATCH_SIZE=16
MAX_EPOCHS="${MAX_EPOCHS:-80}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
BATCH_SIZE="${BATCH_SIZE:-8}"
PROTOCOL="linear"
WEIGHT_INIT="random_init"

WANDB_PROJECT="hsi-bench"
DISK_LOG_ROOT="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/niklas/experiments/hsi-bench"

SLURM_EXCLUDE='\"hpdar04c04s02,hpdar04c02s05,hpdar04c01s01,hpdar04c01s02,hpdar04c01s03,hpdar04c03s04,hpdar04c03s05,hpdar04c05s03,hpdar04c05s05,hpdar04c06s02,hpdar04c06s01\"'
SLURM_ACCOUNT="${SLURM_ACCOUNT:-hpda-c}"
SLURM_TIMEOUT_MIN="${SLURM_TIMEOUT_MIN:-}"

DEBUG_MODE="${DEBUG_MODE:-false}"
FORCE_RESUBMIT="${FORCE_RESUBMIT:-false}"

# --- Precompute per-condition overrides (once) -------------------------------
# Semantic groups (rgb/vis/vnir/swir) and random subsets (rand_*) come from
# two different generator scripts -- both emit the same
# "<sensor>\t<name>\t<band_count>\t<common>\t<wl_nm>\t<wl_um>\t<bw_nm>" shape,
# keyed here by "${sensor}_${name}" in one shared map so submit_job() doesn't
# need to know which source a given condition came from. wl_nm/wl_um/bw_nm
# are resolved but unused below (vit_b takes no wave_list/bandwidth_list --
# see header) -- read into the shared map anyway for uniformity with the
# other ablation scripts' parsing loop.
echo "Resolving band-condition overrides..."
declare -A band_count_map
declare -A band_common_map
while IFS=$'\t' read -r sensor name count common wl_nm wl_um bw_nm; do
  band_count_map["${sensor}_${name}"]="$count"
  band_common_map["${sensor}_${name}"]="$common"
done < <(python3 "${REPO_ROOT}/scripts/downstream/print_band_group_overrides.py")
while IFS=$'\t' read -r sensor name count common wl_nm wl_um bw_nm; do
  band_count_map["${sensor}_${name}"]="$count"
  band_common_map["${sensor}_${name}"]="$common"
done < <(python3 "${REPO_ROOT}/scripts/downstream/print_random_band_overrides.py")

for sensor in emit enmap; do
  for condition in "${CONDITIONS[@]}"; do
    if [ -z "${band_common_map[${sensor}_${condition}]:-}" ]; then
      echo "Error: no band-condition override resolved for sensor=${sensor} condition=${condition}." >&2
      exit 1
    fi
  done
done
echo "Band-condition overrides resolved for: ${!band_common_map[@]}"

# --- Helpers -----------------------------------------------------------------

has_completed_run() {
  local task="$1"
  local condition="$2"
  local sensor="${task_sensor_map[$task]}"
  local experiment_setup="${task}_mlpfusion_linear"
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
  local experiment_setup="${task}_mlpfusion_linear"

  local band_count="${band_count_map[${sensor}_${condition}]}"
  local band_overrides="${band_common_map[${sensor}_${condition}]}"
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
  cmd+=" model.model_type=multi_tap_seg"
  cmd+=" model.freeze_backbone=true"
  cmd+=" model.learning_rate=${LEARNING_RATE}"
  cmd+=" model.weight_decay=${WEIGHT_DECAY}"
  cmd+=" model.backbone_config.patch_size=${PATCH_SIZE}"
  cmd+=" model.backbone_config.img_size=${IMG_SIZE}"
  # Existing field (vit_b.yaml declares in_chans: ${sensor.num_bands} by
  # default) -- plain override, no leading '+' needed. This is the ONLY
  # mechanism vit_b has for becoming band-count-agnostic; see header.
  cmd+=" model.backbone_config.in_chans=${band_count}"
  cmd+=" data.img_size=${IMG_SIZE}"
  cmd+=" data.batch_size=${BATCH_SIZE}"
  cmd+=" trainer=gpu"
  cmd+=" trainer.max_epochs=${MAX_EPOCHS}"
  cmd+=" paths=default"
  cmd+=" paths.log_dir=${DISK_LOG_ROOT}"
  cmd+=" logger.wandb.project=${WANDB_PROJECT}"
  cmd+=" +logger.wandb.name=${task}_${BACKBONE}_mlpfusion_${condition}"
  cmd+=" task_name=${task_name}"
  cmd+=" hydra/launcher=slurm_one_gpu"
  cmd+=" hydra.launcher.exclude=${SLURM_EXCLUDE}"
  cmd+=" hydra.launcher.account=${SLURM_ACCOUNT}"
  if [ -n "$SLURM_TIMEOUT_MIN" ]; then
    cmd+=" hydra.launcher.timeout_min=${SLURM_TIMEOUT_MIN}"
  fi
  cmd+=" hydra.sweep.subdir=${BACKBONE}_${condition}"
  # Same normalization vit_b's full-band baseline already uses (see
  # run_pangaea_consistency_mlp_fusion.sh's build_norm_overrides) -- generic
  # sensor-aware normalization, not bandwise stats keyed to a pretrained
  # encoder (there is none here).
  cmd+=" data.apply_input_normalization=true data.standardize=false data.standardization_stats_path=null"
  # multi_tap_is_vit=true, no explicit tap_indices -- matches how vit_b's
  # full-band baseline leaves taps unforced (12-block ViT-B, multi_tap_seg's
  # own auto-default already applies) -- see
  # run_pangaea_consistency_mlp_fusion.sh's tap_indices_map comment.
  cmd+=" model.multi_tap_is_vit=true"
  cmd+=" ${band_overrides}"
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
  local submit_log="${REPO_ROOT}/scripts/downstream/logs/${task}_${BACKBONE}_mlpfusion_${condition}_submit.log"
  ( eval "$cmd" > "${submit_log}" 2>&1 ; echo "[${task}/${condition}] submission client exited with code $?" >> "${submit_log}" ) &
  echo "Backgrounded (client log: ${submit_log})"
  sleep 2
}

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
