#!/bin/bash
# -----------------------------------------------------------------------------
# Same tasks, same 7 backbones, same hyperparameters as
# run_pangaea_consistency_cdl.sh, but with the MLPFusionHead decoder
# (src/decoders/mlp_fusion_head.py -- faithful port of pangaea-bench's
# pangaea/decoders/seg_mlp_fusion.py: MLPFusionDecoder) instead of UPerNet,
# using the default bilinear upsample_mode (matches pangaea's
# configs/decoder/seg_mlp_fusion.yaml default; the pixel_shuffle/
# transposed_conv options MLPFusionHead also supports are not used here).
#
# DOFA-B, DOFAv2-B, Copernicus-FM-B, Panopticon-B, CARL, SpecAware-B, a
# random-init frozen plain ViT-B/16 baseline (vit_b), and the same backbone fully supervised
# end-to-end (vit_b_finetune) -- all run across every TASKS entry below.
# AdamW, lr=1e-4, weight_decay=0.05, batch_size=8, 80 epochs -- identical to
# the UPerNet script.
#
# Tasks: cdl, bdforet, bnetd, eurocrops, nlcd, treemap (all sensor=enmap),
# plus desis_cdl (sensor=desis) and eo1_cdl (sensor=eo1).
#
# Decoder wiring differs from UPerNet: MLPFusionHead goes through
# model_type=multi_tap_seg (MultiTapSegmentor), not upernet_seg. Its
# in_channels_list is auto-inferred from the backbone (same
# infer_multitap_decoder_info probe lightweight_multitap_head uses -- see
# semantic_segmentation_module.py's is_mlp_fusion branch), but which BLOCKS
# get tapped is controlled by model.tap_indices, not model.decoder_config.
# Forced explicitly per backbone (build_tap_overrides below) to match the
# exact same taps the UPerNet script used per backbone -- so decoder choice
# is the only thing that changes between the two scripts' results for a
# given (task, backbone), not incidentally also which backbone layers get
# read. Without this, multi_tap_seg's own auto-default taps
# (_default_vit_taps_from_backbone) would pick DIFFERENT indices for
# non-12-block backbones (e.g. CARL's 8-block spatial encoder would default
# to [2,5,7], not the [1,3,5,7] pangaea's own carl.yaml/carl_imagewise use).
#
# Already-completed (task, backbone) combinations are skipped automatically
# (checked via a saved checkpoint file, not SLURM exit status -- see
# has_completed_run() below and its comment for why). Set FORCE_RESUBMIT=true
# to resubmit anyway.
#
# Deliberate divergences from pangaea-bench, agreed on beforehand (same as
# the UPerNet script):
#   - Scheduler stays CosineAnnealingLR (his-bench's only wired-in option).
#     Pangaea uses MultiStepLR(milestones=[.6,.9] of total iters, gamma=0.1).
#   - CARL runs with data.apply_input_normalization=false (imagewise-only),
#     matching pangaea's carl_imagewise (image_norm=true) + seg_default_no_norm.
#   - All backbones run at data.img_size=224.
#   - DOFA-B/DOFAv2-B/Copernicus-FM-B are base-only (not -L).
#   - Weight loading is backbone-specific and NOT interchangeable -- see
#     run_pangaea_consistency_cdl.sh's header for the exact mechanism per
#     backbone (identical here, unchanged).
#
# Prerequisites (not automated by this script): same checkpoints as
# run_pangaea_consistency_cdl.sh -- see that script's header for download
# commands. Assumes they're already downloaded (this script doesn't
# re-download).
#
# Usage:
#   bash scripts/downstream/run_pangaea_consistency_mlp_fusion.sh
#   BACKBONES_CSV=carl DEBUG_MODE=true bash scripts/downstream/run_pangaea_consistency_mlp_fusion.sh
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

DECODER="mlp_fusion_head"
IMG_SIZE=224
MAX_EPOCHS="${MAX_EPOCHS:-80}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
BATCH_SIZE="${BATCH_SIZE:-8}"
PROTOCOL="linear"

# Same project as the UPerNet script -- wandb runs are distinguished by name
# (includes "_mlpfusion" suffix, see submit_backbone) and by group/task_name
# (experiment_setup includes "_mlpfusion_linear", not "_upernet_linear").
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

# configs/hydra/launcher/slurm_one_gpu.yaml defaults timeout_min to 600 (10h).
# The UPerNet script's nlcd/panopticon_b and nlcd/carl runs both hit that wall
# clock (NLCD has a much bigger per-epoch dataset than the other tasks).
# MLPFusionHead is lighter than UPerNet (no PSP/FPN), but the backbone forward
# pass + data loading dominate either way, so the same risk likely applies.
# Empty by default (keep the launcher's own 600min); set per-invocation if a
# run times out, e.g. SLURM_TIMEOUT_MIN=1200 TASKS_CSV=nlcd BACKBONES_CSV=panopticon_b,carl.
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
  # configs/backbone/vit_b.yaml defaults patch_size to 4 (tuned for smaller
  # sensor.img_size elsewhere in this repo); overridden to 16 here to match
  # pangaea-bench's vit_scratch.yaml at input_size=224 (196 tokens, standard
  # ViT-B/16) instead of an unnecessarily heavy 56x56=3136-token grid.
  ["vit_b"]=16
  ["vit_b_finetune"]=16
  # Same plain ViT as vit_b, patch_size=8 instead of 16 -- a second
  # random-init frozen baseline point.
  ["vit_b_p8"]=8
  # configs/backbone/spec_vit_b.yaml defaults patch_size to 4; both variants
  # requested explicitly.
  ["spec_vit_b_p8"]=8
  ["spec_vit_b_p16"]=16
)

# vit_b_finetune/vit_b_p8/spec_vit_b_p8/spec_vit_b_p16 all reuse another
# key's Hydra config (vit_b.yaml or spec_vit_b.yaml) -- distinct BACKBONES
# entries only so each gets its own task_name/wandb-group/output-dir and its
# own patch_size/freeze_backbone override. backbone_config_map resolves a
# script-level key to the actual Hydra config group name; unset entries fall
# back to using the key itself (see ${backbone_config_map[$backbone]:-$backbone}
# below).
declare -A backbone_config_map=(
  ["vit_b_finetune"]="vit_b"
  ["vit_b_p8"]="vit_b"
  ["spec_vit_b_p8"]="spec_vit_b"
  ["spec_vit_b_p16"]="spec_vit_b"
)

# freeze_backbone_map: unset entries default to "true" (linear probe, the
# protocol every other backbone in this script uses). vit_b_finetune is the
# one exception -- full end-to-end supervised training from random init.
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
# happen to reuse the identical desis/eo1 stats dirs; only enmap has
# separate per-family stats.
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

# Per-backbone tap indices for model.tap_indices, matching exactly what the
# UPerNet script used per backbone (its decoder output_layers overrides /
# defaults) -- see the header comment for why this needs to be forced rather
# than left to multi_tap_seg's own auto-default.
declare -A tap_indices_map=(
  ["dofa_b"]="[3,5,7,11]"
  ["dofav2_b"]="[4,6,10,11]"
  ["copernicus_fm_b"]="[3,5,7,11]"
  ["panopticon_b"]="[3,5,7,11]"
  ["carl"]="[1,3,5,7]"
  ["specaware_b"]="[3,5,7,11]"
  # vit_b / vit_b_finetune: no entry -- 12-block ViT-B, multi_tap_seg's own
  # auto-default ([3,7,11]) already matches what upernet_head's default
  # [3,5,7,11]-minus-index-5's *effective* taps land close to; left
  # unforced since vit_b never got an explicit override in the UPerNet
  # script either (it used upernet_head.yaml's plain default).
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
      # Random-init baseline -- no checkpoint (neither vit_b.yaml nor
      # spec_vit_b.yaml expose a pretrained-weights field). Nothing to override.
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
      # NOT model.pretrained_weights -- see run_pangaea_consistency_cdl.sh's
      # header for why the generic loader silently fails for this
      # checkpoint's key layout.
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
  local sensor="$2"
  case "$backbone" in
    dofa_b|dofav2_b|copernicus_fm_b)
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
    specaware_b)
      # See run_pangaea_consistency_cdl.sh's build_norm_overrides for why:
      # SpecAware has no built-in physical-value normalization (its
      # patch_embed.scaler only rescales internal projected features), so it
      # needs the datamodule's generic sensor-aware normalization.
      echo "data.apply_input_normalization=true data.standardize=false data.standardization_stats_path=null"
      ;;
    vit_b|vit_b_finetune|vit_b_p8|spec_vit_b_p8|spec_vit_b_p16)
      echo "data.apply_input_normalization=true data.standardize=false data.standardization_stats_path=null"
      ;;
  esac
}

build_tap_overrides() {
  local backbone="$1"
  # multi_tap_is_vit=true: all backbones here are flat ViT-style, never
  # Hiera-like hierarchical -- forced explicitly rather than relying on
  # _backbone_looks_hierarchical's auto-detection, matching the established
  # convention in scripts/downstream/run_competitor_benchmarks.sh
  # ("Competitor backbones are ViT-style in this script").
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
  local protocol="${protocol_map[$backbone]:-$PROTOCOL}"
  local task_dir="${DISK_LOG_ROOT}/downstream/${experiment_setup}/${sensor}/${DECODER}/${backbone}/${protocol}/${weight_init}"

  # See run_pangaea_consistency_cdl.sh's has_completed_run for why this
  # checks the "Best ckpt path:" log line rather than checkpoint existence
  # or sacct/squeue state.
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

  ensure_offline_assets "$backbone" || {
    echo "Warning: failed to prefetch offline assets for ${backbone}. Skipping."
    return 1
  }

  local weight_overrides norm_overrides tap_overrides
  weight_overrides=$(build_weight_overrides "$backbone") || return 1
  norm_overrides=$(build_norm_overrides "$backbone" "$sensor") || return 1
  tap_overrides=$(build_tap_overrides "$backbone")

  local patch_size="${patch_size_map[$backbone]}"
  local img_size_field="${img_size_field_map[$backbone]}"
  local weight_init="${weight_init_tag_map[$backbone]}"
  local protocol="${protocol_map[$backbone]:-$PROTOCOL}"
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
  # UPerNet uses model_type=upernet_seg; MLPFusionHead goes through the
  # multi_tap_seg -> MultiTapSegmentor path instead (same one
  # lightweight_multitap_head uses) -- see header comment.
  cmd+=" model.model_type=multi_tap_seg"
  cmd+=" model.freeze_backbone=${freeze_backbone}"
  cmd+=" model.learning_rate=${LEARNING_RATE}"
  cmd+=" model.weight_decay=${WEIGHT_DECAY}"
  cmd+=" model.backbone_config.patch_size=${patch_size}"
  cmd+=" model.backbone_config.${img_size_field}=${IMG_SIZE}"
  cmd+=" data.img_size=${IMG_SIZE}"
  cmd+=" data.batch_size=${BATCH_SIZE}"
  # See run_pangaea_consistency_cdl.sh's header for why trainer=gpu is
  # required (slurm_one_gpu only allocates 1 GPU; the experiment configs
  # default to trainer=ddp/4 devices).
  cmd+=" trainer=gpu"
  cmd+=" trainer.max_epochs=${MAX_EPOCHS}"
  cmd+=" paths=default"
  cmd+=" paths.log_dir=${DISK_LOG_ROOT}"
  cmd+=" logger.wandb.project=${WANDB_PROJECT}"
  # '+' required: `name` is commented out in configs/logger/wandb.yaml (not
  # present in the config struct at all), so a plain override errors with
  # "Key 'name' is not in struct". "_mlpfusion" suffix (not just
  # ${task}_${backbone}) so these runs don't share a name with the UPerNet
  # script's runs of the same task/backbone in the same flat wandb project.
  cmd+=" +logger.wandb.name=${task}_${backbone}_mlpfusion"
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
  echo "===== Task: ${task} | Backbone: ${backbone} ====="
  echo "$cmd"

  if [ "$DEBUG_MODE" = "true" ]; then
    return 0
  fi

  # Hydra's submitit launcher blocks synchronously on job completion, so run
  # it in the background per (task, backbone) -- see
  # run_pangaea_consistency_cdl.sh's header for why.
  mkdir -p "${REPO_ROOT}/scripts/downstream/logs"
  local submit_log="${REPO_ROOT}/scripts/downstream/logs/${task}_${backbone}_mlpfusion_submit.log"
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
