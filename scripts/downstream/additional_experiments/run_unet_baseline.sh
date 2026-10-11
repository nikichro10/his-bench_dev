#!/bin/bash

# -----------------------------------------------------------------------------
# UNet fully-supervised baseline, across all 9 downstream tasks in the
# pangaea-consistency suite. Mirrors run_pangaea_consistency_cdl.sh /
# run_oxhyperminerals_upernet.sh's protocol (AdamW, lr=1e-4, weight_decay=0.05,
# batch_size=8, 80 epochs, img_size=224) but with a single, standalone
# backbone (unet_b + unet_head) instead of a loop over pretrained backbones:
# a classic 4-level UNet (topology [64,128,256,512]) trained from scratch,
# no pretrained weights -- matches pangaea-bench_dev's unet_encoder + seg_unet
# baseline (fully supervised, finetune=True), NOT another decoder head on top
# of a frozen foundation-model encoder.
#
# Tasks: cdl, bdforet, bnetd, eurocrops, nlcd, treemap (sensor=enmap),
# desis_cdl (sensor=desis), eo1_cdl (sensor=eo1), plus
# oxhyperminerals_emit_l2a (sensor=emit, multilabel -- handled by the same
# `data=oxhyperminerals_emit_l2a_10pct` + `model.num_classes=3` overrides
# run_oxhyperminerals_upernet.sh uses; the multilabel Lightning module is
# selected by the `experiment=downstream/oxhyperminerals_emit_l2a/base`
# config itself, not by anything on this script's command line).
#
# No pretrained-weight loading, no offline-asset prefetching, no
# architecture-specific normalization stats: UNet uses the datamodule's
# generic physical-scale normalization (data.apply_input_normalization=true,
# same choice run_pangaea_consistency_cdl.sh uses for its own
# architecture-agnostic backbones, vit_b/specaware_b).
#
# Deliberate divergence from pangaea-bench, same as the rest of this
# benchmark suite: LR scheduler stays CosineAnnealingLR
# (semantic_segmentation_module.py's only wired-in option), not PANGAEA's
# MultiStepLR.
#
# Already-completed tasks are skipped automatically (checked via the
# "Best ckpt path:" log line, not SLURM exit status -- see has_completed_run()
# below). Set FORCE_RESUBMIT=true to resubmit anyway.
#
# Usage:
#   bash scripts/downstream/run_unet_baseline.sh
#   TASKS_CSV=cdl DEBUG_MODE=true bash scripts/downstream/run_unet_baseline.sh
# -----------------------------------------------------------------------------
set -e

eval "$(micromamba shell hook --shell bash)"
micromamba activate hsi-bench

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

REPO_ROOT="$(pwd -P)"

# --- Fixed experiment configuration -----------------------------------------
TASKS=(cdl bdforet bnetd eurocrops nlcd treemap desis_cdl eo1_cdl oxhyperminerals_emit_l2a)
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
  ["oxhyperminerals_emit_l2a"]="emit"
)

BACKBONE="unet_b"
DECODER="unet_head"
IMG_SIZE=224
MAX_EPOCHS="${MAX_EPOCHS:-80}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
BATCH_SIZE="${BATCH_SIZE:-8}"
PROTOCOL="finetune"          # fully supervised, not linear-probe
WEIGHT_INIT="random_init_finetune"
FREEZE_BACKBONE="false"

# Distinct experiment_setup from the upernet-consistency suite (different
# decoder/model_type entirely) -- keeps the disk layout self-describing:
# downstream/<task>_unet_baseline/<sensor>/unet_head/unet_b/finetune/random_init_finetune/
EXPERIMENT_SETUP_SUFFIX="unet_baseline"

WANDB_PROJECT="hsi-bench"
DISK_LOG_ROOT="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/niklas/experiments/hsi-bench"

SLURM_EXCLUDE='\"hpdar04c04s02,hpdar04c02s05,hpdar04c01s01,hpdar04c01s02,hpdar04c01s03,hpdar04c03s04,hpdar04c03s05,hpdar04c05s03,hpdar04c05s05,hpdar04c06s02\"'
SLURM_ACCOUNT="${SLURM_ACCOUNT:-hpda-c}"
SLURM_TIMEOUT_MIN="${SLURM_TIMEOUT_MIN:-}"

DEBUG_MODE="${DEBUG_MODE:-false}"
FORCE_RESUBMIT="${FORCE_RESUBMIT:-false}"

# --- Helpers -----------------------------------------------------------------

has_completed_run() {
  local task="$1"
  local sensor="${task_sensor_map[$task]}"
  local experiment_setup="${task}_${EXPERIMENT_SETUP_SUFFIX}"
  local task_dir="${DISK_LOG_ROOT}/downstream/${experiment_setup}/${sensor}/${DECODER}/${BACKBONE}/${PROTOCOL}/${WEIGHT_INIT}"

  # See run_pangaea_consistency_cdl.sh's has_completed_run() for why this is
  # the only reliable completion signal (checkpoint existence and SLURM exit
  # status are both misleading here).
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
  local experiment_setup="${task}_${EXPERIMENT_SETUP_SUFFIX}"
  local task_name="downstream/${experiment_setup}/${sensor}/${DECODER}/${BACKBONE}/${PROTOCOL}/${WEIGHT_INIT}"

  local data_overrides="" num_classes_override=""
  if [ "$task" = "oxhyperminerals_emit_l2a" ]; then
    data_overrides="data=oxhyperminerals_emit_l2a_10pct"
    num_classes_override="model.num_classes=3"
  fi

  local cmd="python src/train.py --config-name=downstream -m"
  cmd+=" experiment=downstream/${task}/base"
  cmd+=" ${data_overrides}"
  cmd+=" backbone@model.backbone_config=${BACKBONE}"
  cmd+=" decoder@model.decoder_config=${DECODER}"
  cmd+=" sensor=${sensor}"
  cmd+=" sensor@data.sensor_config=${sensor}"
  cmd+=" experiment_setup=${experiment_setup}"
  cmd+=" protocol=${PROTOCOL}"
  cmd+=" weight_init=${WEIGHT_INIT}"
  cmd+=" backbone_name=${BACKBONE}"
  cmd+=" decoder_name=${DECODER}"
  cmd+=" model.model_type=unet_seg"
  cmd+=" ${num_classes_override}"
  cmd+=" model.freeze_backbone=${FREEZE_BACKBONE}"
  cmd+=" model.learning_rate=${LEARNING_RATE}"
  cmd+=" model.weight_decay=${WEIGHT_DECAY}"
  cmd+=" model.backbone_config.img_size=${IMG_SIZE}"
  cmd+=" data.img_size=${IMG_SIZE}"
  cmd+=" data.batch_size=${BATCH_SIZE}"
  # Generic physical-scale normalization -- same choice run_pangaea_consistency_cdl.sh
  # uses for architecture-agnostic backbones with no dedicated stats file (vit_b, specaware_b).
  cmd+=" data.apply_input_normalization=true data.standardize=false data.standardization_stats_path=null"
  # trainer=ddp is cdl/base.yaml's default (devices: 4); slurm_one_gpu only
  # allocates 1 GPU per job -- see run_pangaea_consistency_cdl.sh's comment.
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
