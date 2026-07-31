#!/bin/bash
# -----------------------------------------------------------------------------
# DOFA-B, DOFAv2-B, Copernicus-FM-B, Panopticon-B, CARL, a random-init frozen
# plain ViT-B/16 baseline (vit_b -- configs/backbone/vit_b.yaml,
# TimmViTBackbone, pretrained=false; the his-bench equivalent of
# pangaea-bench's vit_scratch.yaml + finetune=False "lower-bound baseline
# showing how much performance comes from the decoder alone with random
# features"), and the same backbone fully supervised end-to-end
# (vit_b_finetune -- freeze_backbone=false, protocol=finetune; equivalent of
# vit_scratch.yaml + finetune=True) -- all run across every TASKS entry below,
# with the UPerNet decoder. Everywhere except backbone-specific weight
# loading/normalization/decoder taps, hyperparameters and setup are IDENTICAL
# across tasks and backbones, matched to pangaea-bench's segmentation
# linear-probe defaults (configs/train.yaml + configs/task/segmentation.yaml +
# configs/optimizer/adamw.yaml in pangaea-bench_dev):
#   AdamW, lr=1e-4, weight_decay=0.05, batch_size=8, 80 epochs.
#
# Tasks: cdl, bdforet, bnetd, eurocrops, nlcd, treemap (all sensor=enmap),
# plus desis_cdl (sensor=desis) and eo1_cdl (sensor=eo1) -- the two
# cross-sensor tasks, which is why normalization stats are looked up
# per-sensor (dofa_family_stats_path_map / panopticon_stats_path_map below),
# not hardcoded to the enmap paths the way the original cdl-only version of
# this script had them.
#
# Not yet added (blocked or bigger lifts, see conversation): h2sr,
# gaofen5_wuhan, hyperview, oxhyperminerals_emit_l2a -- data not present at
# the default path profile for any of the four; hyperview is also regression
# (needs RegUPerNet, which has a known unfixed bug -- self.encoder referenced
# but never set when in_channels=None) and corine is multilabel
# classification (different model_type/decoder pattern) -- neither wired
# into this script yet.
#
# Already-completed (task, backbone) combinations are skipped automatically
# (checked via a saved checkpoint file, not SLURM exit status -- see
# has_completed_run() below and its comment for why). Set FORCE_RESUBMIT=true
# to resubmit anyway.
#
# Deliberate divergences from pangaea-bench, agreed on beforehand:
#   - Scheduler stays CosineAnnealingLR (his-bench's only wired-in option,
#     src/models/semantic_segmentation_module.py:570-577). Pangaea uses
#     MultiStepLR(milestones=[.6,.9] of total iters, gamma=0.1) instead — a
#     step schedule, not a smooth decay. Not reproduced here by choice.
#   - CARL runs with data.apply_input_normalization=false (imagewise-only,
#     relying purely on CARL's internal per-image standardize) — the
#     equivalent of pangaea's carl_imagewise (image_norm=true) +
#     seg_default_no_norm pairing, NOT his-bench's own competitor-benchmark
#     default of apply_input_normalization=true (which clips to [0,1] and
#     would double-normalize on top of CARL's internal standardize).
#   - CARL's decoder output_layers is overridden to [1,3,5,7] (its spatial
#     encoder is truncated to 8 blocks; upernet_head's default [3,5,7,11]
#     would index out of range). Matches pangaea's carl(.yaml)/carl_imagewise
#     output_layers exactly.
#   - All three backbones run at data.img_size=224 (DOFA/Panopticon already
#     default there; CARL's his-bench competitor-script default is 128 —
#     overridden here to match pangaea's carl.yaml input_size=224).
#   - DOFA-B only (not DOFA-L). DOFAv2/Copernicus-FM likewise base-only.
#   - DOFAv2-B and Copernicus-FM-B use their own paper-recommended UPerNet
#     taps ([4,6,10,11] and [3,5,7,11] respectively), not upernet_head's
#     generic [3,5,7,11] default -- matches pangaea's dofav2.yaml /
#     copernicus_fm_base.yaml exactly.
#   - Weight loading is backbone-specific and NOT interchangeable:
#       * dofa_b, dofav2_b:  model.pretrained_weights=<ckpt>  (goes through
#         his-bench's generic state_dict loader in model_utils.py, which does
#         a plain name-for-name match -- works because both checkpoints'
#         keys already align with these encoders' own attribute names).
#       * copernicus_fm_b:   model.backbone_config.encoder_weights=<ckpt>
#         (loaded inside the encoder's own __init__, NOT the generic loader).
#         Required: the official Copernicus-FM checkpoint stores the patch
#         embedding under "patch_embed_spectral.*" (flat, unprefixed) while
#         this encoder's own weights live under "model.patch_embed.*" -- the
#         generic loader has no idea about that rename and would silently
#         leave the entire model randomly initialized (verified empirically
#         before wiring this in: 100% of params came back unmatched).
#       * panopticon_b: model.backbone_config.checkpoint_path=<ckpt>
#       * carl:         model.backbone_config.ssl_ckpt_path=<ckpt>
#     (all four self-load or generic-load correctly for their own case; don't
#     swap the mechanism between backbones without re-verifying the same way.)
#
# Prerequisites (not automated by this script):
#   1. Download the checkpoints:
#        mkdir -p pretrained_models/comparison/{dofa_b,dofav2_b,copernicus_fm_b,panopticon,carl}
#        wget -O pretrained_models/comparison/dofa_b/DOFA_ViT_base_e100.pth \
#          https://huggingface.co/XShadow/DOFA/resolve/main/DOFA_ViT_base_e100.pth
#        wget -O pretrained_models/comparison/dofav2_b/dofav2_vit_base_e150.pth \
#          https://huggingface.co/XShadow/DOFA/resolve/main/dofav2_vit_base_e150.pth
#        wget -O pretrained_models/comparison/copernicus_fm_b/CopernicusFM_ViT_base_varlang_e100.pth \
#          https://huggingface.co/wangyi111/Copernicus-FM/resolve/main/CopernicusFM_ViT_base_varlang_e100.pth
#        wget -O pretrained_models/comparison/panopticon/panopticon_vitb14_teacher.pth \
#          https://huggingface.co/lewaldm/panopticon/resolve/main/panopticon_vitb14_teacher.pth
#        wget -O pretrained_models/comparison/carl/carl_ssl_checkpoint.ckpt \
#          https://zenodo.org/records/18671944/files/ssl_checkpoint_carl.ckpt
#   2. CDL data present at ${paths.data_dir}/cdl (already the case on this
#      system's default path profile).
#
# Usage:
#   bash scripts/downstream/run_pangaea_consistency_cdl.sh
#   BACKBONES_CSV=carl DEBUG_MODE=true bash scripts/downstream/run_pangaea_consistency_cdl.sh
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

REPO_ROOT="$(pwd -P)"
OFFLINE_ASSET_ROOT="${OFFLINE_ASSET_ROOT:-${REPO_ROOT}/pretrained_models/comparison/offline_assets}"
HF_HOME_SHARED="${HF_HOME_SHARED:-${OFFLINE_ASSET_ROOT}/hf_home}"
HF_HUB_CACHE_SHARED="${HF_HUB_CACHE_SHARED:-${HF_HOME_SHARED}/hub}"
TRANSFORMERS_CACHE_SHARED="${TRANSFORMERS_CACHE_SHARED:-${HF_HOME_SHARED}/transformers}"
TORCH_HOME_SHARED="${TORCH_HOME_SHARED:-${OFFLINE_ASSET_ROOT}/torch_home}"
PANOPTICON_LOCAL_REPO="${PANOPTICON_LOCAL_REPO:-${OFFLINE_ASSET_ROOT}/panopticon_repo}"
CARL_PREFETCH_MARKER="${OFFLINE_ASSET_ROOT}/.carl_prefetched"
PANOPTICON_PREFETCH_MARKER="${OFFLINE_ASSET_ROOT}/.panopticon_prefetched"

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

# Source sensor per task -- read directly from each
# configs/experiment/downstream/<task>/base.yaml's `override /sensor:` line.
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

DECODER="upernet_head"
IMG_SIZE=224
MAX_EPOCHS="${MAX_EPOCHS:-80}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
BATCH_SIZE="${BATCH_SIZE:-8}"
PROTOCOL="linear"

# New project, doesn't exist yet — wandb creates it automatically on first log.
WANDB_PROJECT="hsi-bench"

# Dedicated disk location, sibling to pangaea-bench_dev's per-benchmark folders
# under the same niklas/experiments/ root (e.g. carl_dataset_norm_seg, geobench).
DISK_LOG_ROOT="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/niklas/experiments/hsi-bench"

# Node exclude list copied verbatim from pangaea-bench_dev's
# scripts/slurm_scripts/run_carl_seg_benchmark.sh (same hpda2 cluster).
# Wrapped in escaped quotes: under `-m`/multirun, Hydra's override parser
# treats a bare comma-separated value as a sweep list unless it's quoted as
# a single literal string (confirmed via hydra.core.override_parser).
SLURM_EXCLUDE='\"hpdar04c04s02,hpdar04c02s05,hpdar04c01s01,hpdar04c01s02,hpdar04c01s03,hpdar04c03s04,hpdar04c03s05,hpdar04c05s03,hpdar04c05s05,hpdar04c06s02\"'

# configs/hydra/launcher/slurm_one_gpu.yaml defaults to account=pn52pi-c, which
# isn't valid for this user (confirmed via `sacctmgr show associations
# user=di54zop`: only hpda-c and lowprio are). Override rather than editing
# the shared launcher config, since other users of this repo may rely on the
# pn52pi-c default.
SLURM_ACCOUNT="${SLURM_ACCOUNT:-hpda-c}"

BACKBONES=(dofa_b dofav2_b copernicus_fm_b panopticon_b carl vit_b vit_b_finetune)
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
)
declare -A weight_init_tag_map=(
  ["dofa_b"]="dofa_pretrained"
  ["dofav2_b"]="dofav2_pretrained"
  ["copernicus_fm_b"]="copernicus_fm_pretrained"
  ["panopticon_b"]="panopticon_pretrained"
  ["carl"]="carl_pretrained"
  ["vit_b"]="random_init"
  ["vit_b_finetune"]="random_init_finetune"
)
declare -A img_size_field_map=(
  ["dofa_b"]="input_size"
  ["dofav2_b"]="input_size"
  ["copernicus_fm_b"]="input_size"
  ["panopticon_b"]="img_size"
  ["carl"]="img_size"
  ["vit_b"]="img_size"
  ["vit_b_finetune"]="img_size"
)
declare -A patch_size_map=(
  ["dofa_b"]=16
  ["dofav2_b"]=14
  ["copernicus_fm_b"]=16
  ["panopticon_b"]=14
  ["carl"]=8
  # configs/backbone/vit_b.yaml defaults patch_size to 4 (tuned for smaller
  # sensor.img_size elsewhere in this repo); overridden to 16 here to match
  # pangaea-bench's vit_scratch.yaml at input_size=224 (196 tokens, standard
  # ViT-B/16) instead of an unnecessarily heavy 56x56=3136-token grid.
  ["vit_b"]=16
  ["vit_b_finetune"]=16
)

# vit_b_finetune is the same architecture/config as vit_b (configs/backbone/
# vit_b.yaml) -- it's a distinct BACKBONES entry only so it gets its own
# task_name/wandb-group/output-dir and its own freeze_backbone setting.
# backbone_config_map resolves a script-level key to the actual Hydra config
# group name; unset entries fall back to using the key itself (see
# ${backbone_config_map[$backbone]:-$backbone} below).
declare -A backbone_config_map=(
  ["vit_b_finetune"]="vit_b"
)

# freeze_backbone_map: unset entries default to "true" (linear probe, the
# protocol every other backbone in this script uses). vit_b_finetune is the
# one exception -- full end-to-end supervised training from random init,
# matching pangaea-bench's vit_scratch + finetune=True (run_vit_scratch_*
# scripts) as opposed to vit_scratch + finetune=False (run_vit_frozen_*,
# what vit_b already reproduces).
declare -A freeze_backbone_map=(
  ["vit_b_finetune"]="false"
)

# protocol_map: unset entries default to $PROTOCOL ("linear"). Purely a
# label used in task_name/wandb-group/disk paths so a finetuned run doesn't
# get filed under ".../linear/..." -- doesn't affect training behavior
# (that's freeze_backbone_map above).
declare -A protocol_map=(
  ["vit_b_finetune"]="finetune"
)

# Per-sensor bandwise normalization stats for the DOFA-family backbones
# (dofa_b, dofav2_b, copernicus_fm_b) and for Panopticon-B. Both families
# happen to reuse the identical desis/eo1 stats dirs (confirmed present:
# data/desis_statistics, data/eo1_statistics -- same files used by
# run_competitor_benchmarks.sh's dofa_stats_path_map/panopticon_stats_path_map
# for these two sensors); only enmap has separate per-family stats.
declare -A dofa_family_stats_path_map=(
  ["enmap"]="data/dofa_statistics"
  ["desis"]="data/desis_statistics"
  ["eo1"]="data/eo1_statistics"
)
declare -A panopticon_stats_path_map=(
  ["enmap"]="data/panopticon_statistics"
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
  if [ "$backbone" = "vit_b" ] || [ "$backbone" = "vit_b_finetune" ]; then
    # Random-init baseline -- no checkpoint. configs/backbone/vit_b.yaml
    # already defaults to pretrained: false; nothing to override.
    return 0
  fi
  local ckpt="${checkpoint_path_map[$backbone]}"
  validate_existing_path "${backbone} checkpoint" "$ckpt" || return 1
  case "$backbone" in
    dofa_b|dofav2_b)
      echo "model.pretrained_weights=${ckpt}"
      ;;
    copernicus_fm_b)
      # NOT model.pretrained_weights -- see the header comment for why the
      # generic loader silently fails for this checkpoint's key layout.
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
      # Same DOFA-family bandwise stats for all three: pangaea-bench uses the
      # identical seg_default_norm_no_pad preprocessing (bandwise
      # NormalizeMeanStd) for dofa/dofav2/copernicus_fm alike -- no
      # architecture-specific stats exist for the v2/Copernicus variants.
      local stats_path="${dofa_family_stats_path_map[$sensor]}"
      if [ -z "$stats_path" ]; then
        echo "Error: no DOFA-family stats path configured for sensor '${sensor}'." >&2
        return 1
      fi
      validate_existing_path "DOFA-family stats (${sensor})" "$stats_path" || return 1
      echo "data.apply_input_normalization=false data.standardize=true data.standardization_mode=bandwise data.standardization_stats_path=${stats_path}"
      ;;
    panopticon_b)
      local stats_path="${panopticon_stats_path_map[$sensor]}"
      if [ -z "$stats_path" ]; then
        echo "Error: no Panopticon stats path configured for sensor '${sensor}'." >&2
        return 1
      fi
      validate_existing_path "Panopticon stats (${sensor})" "$stats_path" || return 1
      echo "data.apply_input_normalization=false data.standardize=true data.standardization_mode=bandwise data.standardization_stats_path=${stats_path}"
      ;;
    carl)
      # Imagewise-only: rely solely on CARL's internal per-image standardization,
      # matching pangaea-bench's carl_imagewise + seg_default_no_norm pairing.
      echo "data.apply_input_normalization=false data.standardize=false data.standardization_stats_path=null"
      ;;
    vit_b|vit_b_finetune)
      # Plain ViT has no wavelength-conditioned patch embed and no
      # architecture-specific stats file -- just the datamodule's generic
      # physical scale+clip normalization (same choice as competitor-script's
      # default for architecture-agnostic backbones like specaware_b).
      echo "data.apply_input_normalization=true data.standardize=false data.standardization_stats_path=null"
      ;;
  esac
}

build_decoder_overrides() {
  local backbone="$1"
  case "$backbone" in
    carl)
      echo "model.decoder_config.output_layers=[1,3,5,7]"
      ;;
    dofav2_b)
      # Matches pangaea's dofav2.yaml -- not upernet_head's generic [3,5,7,11].
      echo "model.decoder_config.output_layers=[4,6,10,11]"
      ;;
    copernicus_fm_b)
      # Matches pangaea's copernicus_fm_base.yaml. Coincides with
      # upernet_head's own default, but set explicitly so it's not relying
      # on that default silently staying in sync.
      echo "model.decoder_config.output_layers=[3,5,7,11]"
      ;;
  esac
}

enable_offline_runtime_env() {
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
}

has_completed_run() {
  local task="$1"
  local backbone="$2"
  local sensor="${task_sensor_map[$task]}"
  local experiment_setup="${task}_upernet_linear"
  local weight_init="${weight_init_tag_map[$backbone]}"
  local protocol="${protocol_map[$backbone]:-$PROTOCOL}"
  local task_dir="${DISK_LOG_ROOT}/downstream/${experiment_setup}/${sensor}/${DECODER}/${backbone}/${protocol}/${weight_init}"

  # A checkpoint FILE EXISTING is not sufficient evidence of completion:
  # ModelCheckpoint (save_top_k=1) overwrites the best-so-far checkpoint
  # continuously throughout training, so one exists from the first
  # validation epoch onward -- caught this directly, treating an
  # actively-RUNNING job (6 minutes in, epoch 12/80) as "completed" because
  # its best-so-far checkpoint already existed on disk.
  #
  # The reliable signal is the "Best ckpt path:" line train.py logs, which
  # only gets printed after trainer.test() runs -- i.e. only after ALL
  # epochs of trainer.fit() finish (train=true, test=true both run
  # sequentially). Also do NOT trust sacct/squeue COMPLETED/exit-code state
  # instead: we've separately seen SLURM report a job COMPLETED after it
  # crashed before a single epoch ran (e.g. the trainer=ddp/1-GPU mismatch
  # this script's trainer=gpu override now prevents) -- this codebase's
  # error wrapping logs the exception but doesn't propagate a nonzero exit
  # code, so exit status alone is not trustworthy either.
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
  local experiment_setup="${task}_upernet_linear"

  ensure_offline_assets "$backbone" || {
    echo "Warning: failed to prefetch offline assets for ${backbone}. Skipping."
    return 1
  }

  local weight_overrides norm_overrides decoder_overrides
  weight_overrides=$(build_weight_overrides "$backbone") || return 1
  norm_overrides=$(build_norm_overrides "$backbone" "$sensor") || return 1
  decoder_overrides=$(build_decoder_overrides "$backbone")

  local patch_size="${patch_size_map[$backbone]}"
  local img_size_field="${img_size_field_map[$backbone]}"
  local weight_init="${weight_init_tag_map[$backbone]}"
  local protocol="${protocol_map[$backbone]:-$PROTOCOL}"
  # backbone_config_map lets a script-level key reuse another key's Hydra
  # config (vit_b_finetune -> vit_b.yaml) while keeping its own task_name/
  # freeze_backbone/protocol identity.
  local backbone_config="${backbone_config_map[$backbone]:-$backbone}"
  local freeze_backbone="${freeze_backbone_map[$backbone]:-true}"
  local task_name="downstream/${experiment_setup}/${sensor}/${DECODER}/${backbone}/${protocol}/${weight_init}"

  local cmd="python src/train.py --config-name=downstream -m"
  cmd+=" experiment=downstream/${task}/base"
  cmd+=" backbone@model.backbone_config=${backbone_config}"
  cmd+=" decoder@model.decoder_config=${DECODER}"
  cmd+=" sensor=${sensor}"
  cmd+=" sensor@data.sensor_config=${sensor}"
  cmd+=" experiment_setup=${experiment_setup}"
  cmd+=" protocol=${protocol}"
  cmd+=" weight_init=${weight_init}"
  cmd+=" backbone_name=${backbone}"
  cmd+=" decoder_name=${DECODER}"
  cmd+=" model.model_type=upernet_seg"
  cmd+=" model.freeze_backbone=${freeze_backbone}"
  cmd+=" model.learning_rate=${LEARNING_RATE}"
  cmd+=" model.weight_decay=${WEIGHT_DECAY}"
  cmd+=" model.backbone_config.patch_size=${patch_size}"
  cmd+=" model.backbone_config.${img_size_field}=${IMG_SIZE}"
  cmd+=" data.img_size=${IMG_SIZE}"
  cmd+=" data.batch_size=${BATCH_SIZE}"
  # cdl/base.yaml defaults to trainer=ddp (devices: 4), but slurm_one_gpu only
  # allocates 1 GPU per job. trainer=gpu is the matching single-device preset
  # (accelerator: gpu, devices: 1, no DDP strategy). Without this override,
  # the run crashes immediately with a device-count MisconfigurationException
  # — and this codebase's error wrapping swallows it, so SLURM still reports
  # the job as COMPLETED even though nothing trained. Always check the actual
  # .submitit log, not just `sacct` exit codes.
  cmd+=" trainer=gpu"
  cmd+=" trainer.max_epochs=${MAX_EPOCHS}"
  cmd+=" paths=default"
  cmd+=" paths.log_dir=${DISK_LOG_ROOT}"
  cmd+=" logger.wandb.project=${WANDB_PROJECT}"
  cmd+=" task_name=${task_name}"
  cmd+=" hydra/launcher=slurm_one_gpu"
  cmd+=" hydra.launcher.exclude=${SLURM_EXCLUDE}"
  cmd+=" hydra.launcher.account=${SLURM_ACCOUNT}"
  cmd+=" hydra.sweep.subdir=${backbone}"
  cmd+=" ${weight_overrides}"
  cmd+=" ${norm_overrides}"
  cmd+=" ${decoder_overrides}"

  echo ""
  echo "===== Task: ${task} | Backbone: ${backbone} ====="
  echo "$cmd"

  if [ "$DEBUG_MODE" = "true" ]; then
    return 0
  fi

  # Hydra's submitit launcher blocks synchronously on job completion
  # (calls .results() after sbatch submission), so run it in the background
  # per (task, backbone) — otherwise this script would sit blocked for the
  # entire training duration of one run before even submitting the next.
  # Client-side submission log (Hydra's own output, not the training job's
  # own stdout/stderr, which submitit writes under <sweep_dir>/.submitit/).
  mkdir -p "${REPO_ROOT}/scripts/downstream/logs"
  local submit_log="${REPO_ROOT}/scripts/downstream/logs/${task}_${backbone}_submit.log"
  ( eval "$cmd" > "${submit_log}" 2>&1 ; echo "[${task}/${backbone}] submission client exited with code $?" >> "${submit_log}" ) &
  echo "Backgrounded (client log: ${submit_log})"
  sleep 2
}

enable_offline_runtime_env

for task in "${TASKS[@]}"; do
  for backbone in "${BACKBONES[@]}"; do
    if [ "$FORCE_RESUBMIT" != "true" ]; then
      existing_ckpt=$(has_completed_run "$task" "$backbone") && {
        echo ""
        echo "===== Task: ${task} | Backbone: ${backbone} ====="
        echo "Skipping: found an existing checkpoint from a completed run:"
        echo "  ${existing_ckpt}"
        echo "(set FORCE_RESUBMIT=true to resubmit anyway)"
        continue
      }
    fi
    submit_backbone "$task" "$backbone" || echo "Warning: ${task}/${backbone} run was not submitted (see error above)."
  done
done

echo ""
echo "All jobs submitted."
