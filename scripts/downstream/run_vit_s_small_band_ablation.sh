#!/bin/bash
# -----------------------------------------------------------------------------
# Phase 1 of the small-band-count random-subset sweep: a supervised, FULLY
# FINETUNED ViT-Small (random init, unfrozen backbone -- not a frozen linear
# probe like every other backbone in this benchmark) trained end-to-end on
# random band subsets of size 3/7/10/15/20. Purpose: sanity-check that these
# small subsets actually carry usable information before spending compute
# re-running the same (seed, size) draws through the frozen pretrained
# encoders (DOFA-B/DOFAv2-B/Copernicus-FM-B/Panopticon-B/CARL/UniverSat-B) in
# a later Phase-2 script. If test performance climbs as band count goes
# 3->7->10->15->20 here, that's real signal, not a frozen-probe artifact --
# since this backbone can fully adapt its weights to whatever the 3-20 given
# bands actually contain.
#
# Why ViT-Small, not ViT-Base (vit_b, already used elsewhere in this
# benchmark as a frozen-probe control): full end-to-end finetuning of a
# random-init ViT-Base on a handful of bands and a comparatively small
# downstream dataset risks overfitting/being needlessly heavy; ViT-Small
# (configs/backbone/vit_s.yaml, timm "vit_small_patch16_224", added this
# session -- TimmViTBackbone is generic over model_name, no src/ changes
# needed) is the lighter, more appropriate choice for this specific
# from-scratch-supervised setting.
#
# Conditions: rand_n3/n7/n10/n15/n20, 3 seeds each = 15 conditions, RANDOM
# ONLY (no semantic-region counterpart -- this is purely a band-COUNT sweep,
# see print_small_band_overrides.py's header). n3/n20 draws are numerically
# identical to the existing "rgb"/"n20" draws from
# print_random_band_overrides.py (same (seed, n) -> same
# random.Random(seed).sample(...)), kept under a fresh "n3"/"n20" label here
# so this sweep reads as self-contained. No "full" (all-bands) condition --
# out of scope for this sanity check.
#
# Tasks: the 3 existing ablation tasks (oxhyperminerals_emit_l2a, eurocrops,
# bnetd) PLUS bdforet (French BD Foret forest-type segmentation, EnMAP,
# 12 classes) -- a second, genuinely different tree/forest task from BNETD's
# Ivorian forest-type map, added this session. bdforet already exists as a
# working task (configs/experiment/downstream/bdforet/base.yaml,
# src/datamodules/bdforet_datamodule.py) and was previously run through this
# exact multi_tap_seg/mlp_fusion_head path via
# run_pangaea_consistency_mlp_fusion.sh -- slots in with zero new plumbing
# (same enmap sensor/stats convention as eurocrops/bnetd, own
# model.num_classes=12 already baked into its base.yaml, no _10pct split
# needed). trainer.max_epochs is forced to 80 here (bdforet's own base.yaml
# defaults to 100) to stay comparable with the other 3 tasks, per explicit
# instruction.
#
# vit_s has NO wave_list field at all (plain timm ViT, no wavelength
# conditioning) -- same as vit_b, see run_vit_b_band_ablation.sh's header.
# The only band-count-agnostic hook is model.backbone_config.in_chans=<N>,
# a plain override of an existing field (no leading '+').
#
# freeze_backbone=false, protocol=finetune, weight_init=random_init_finetune
# -- full supervised finetune, mirroring exactly how vit_b_finetune already
# works elsewhere in this benchmark (run_pangaea_consistency_mlp_fusion.sh).
# No checkpoint to load (random init), so no weight_overrides.
#
# Already-completed (task, condition) combinations are skipped automatically
# via the "Best ckpt path:" log line -- same rule as every other script in
# this benchmark, never squeue/wandb state alone.
#
# Usage:
#   bash scripts/downstream/run_vit_s_small_band_ablation.sh
#   TASKS_CSV=oxhyperminerals_emit_l2a CONDITIONS_CSV=rand_n3_s0 \
#     DEBUG_MODE=true bash scripts/downstream/run_vit_s_small_band_ablation.sh
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
# "Best ckpt path:" log line has_completed_run()/the analysis notebook read.
# Sync later with `wandb sync <run dir>` once connectivity is confirmed back.
export WANDB_MODE=offline

REPO_ROOT="$(pwd -P)"

# --- Fixed experiment configuration -----------------------------------------
TASKS=(oxhyperminerals_emit_l2a eurocrops bnetd bdforet)
TASKS_CSV="${TASKS_CSV:-}"
if [ -n "$TASKS_CSV" ]; then
  IFS=',' read -ra TASKS <<< "$TASKS_CSV"
fi

CONDITIONS=(rand_n3_s0 rand_n3_s1 rand_n3_s2 rand_n7_s0 rand_n7_s1 rand_n7_s2 rand_n10_s0 rand_n10_s1 rand_n10_s2 rand_n15_s0 rand_n15_s1 rand_n15_s2 rand_n20_s0 rand_n20_s1 rand_n20_s2)
CONDITIONS_CSV="${CONDITIONS_CSV:-}"
if [ -n "$CONDITIONS_CSV" ]; then
  IFS=',' read -ra CONDITIONS <<< "$CONDITIONS_CSV"
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

BACKBONE="vit_s"
DECODER="mlp_fusion_head"
IMG_SIZE=224
PATCH_SIZE=16
MAX_EPOCHS="${MAX_EPOCHS:-80}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
BATCH_SIZE="${BATCH_SIZE:-8}"
PROTOCOL="finetune"
WEIGHT_INIT="random_init_finetune"

WANDB_PROJECT="hsi-bench"
DISK_LOG_ROOT="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/niklas/experiments/hsi-bench"

SLURM_EXCLUDE='\"hpdar04c04s02,hpdar04c02s05,hpdar04c01s01,hpdar04c01s02,hpdar04c01s03,hpdar04c03s04,hpdar04c03s05,hpdar04c05s03,hpdar04c05s05,hpdar04c06s02,hpdar04c06s01\"'
SLURM_ACCOUNT="${SLURM_ACCOUNT:-hpda-c}"
SLURM_TIMEOUT_MIN="${SLURM_TIMEOUT_MIN:-}"

DEBUG_MODE="${DEBUG_MODE:-false}"
FORCE_RESUBMIT="${FORCE_RESUBMIT:-false}"

# --- Precompute per-condition overrides (once) -------------------------------
echo "Resolving small-band-count overrides..."
declare -A band_common_map
while IFS=$'\t' read -r sensor condition count common wl_nm wl_um bw_nm; do
  band_common_map["${sensor}_${condition}"]="$common"
done < <(python3 "${REPO_ROOT}/scripts/downstream/print_small_band_overrides.py")

for sensor in emit enmap; do
  for condition in "${CONDITIONS[@]}"; do
    if [ -z "${band_common_map[${sensor}_${condition}]:-}" ]; then
      echo "Error: no small-band override resolved for sensor=${sensor} condition=${condition}." >&2
      exit 1
    fi
  done
done
echo "Small-band overrides resolved for: ${!band_common_map[@]}"

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

  local band_overrides="${band_common_map[${sensor}_${condition}]}"
  # band_overrides already carries model.probe_in_chans -- parse the same
  # count back out for model.backbone_config.in_chans (vit_s's only
  # band-count-agnostic hook, see header).
  local band_count
  band_count=$(echo "$band_overrides" | grep -oE 'probe_in_chans=[0-9]+' | grep -oE '[0-9]+')

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
  cmd+=" model.freeze_backbone=false"
  cmd+=" model.learning_rate=${LEARNING_RATE}"
  cmd+=" model.weight_decay=${WEIGHT_DECAY}"
  cmd+=" model.backbone_config.patch_size=${PATCH_SIZE}"
  cmd+=" model.backbone_config.img_size=${IMG_SIZE}"
  # Existing field (vit_s.yaml declares in_chans: ${sensor.num_bands} by
  # default) -- plain override, no leading '+' needed. Only mechanism vit_s
  # has for becoming band-count-agnostic; see header.
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
  # Generic sensor-aware normalization, not bandwise stats keyed to a
  # pretrained encoder (there is none here) -- same as vit_b's own baseline.
  cmd+=" data.apply_input_normalization=true data.standardize=false data.standardization_stats_path=null"
  # multi_tap_is_vit=true, no explicit tap_indices -- matches vit_b (12-block
  # ViT-style backbone, multi_tap_seg's own auto-default taps apply; ViT-S is
  # narrower than ViT-B but the same depth=12, so the same default holds).
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
