#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/sft_readme_config.sh
source "${SCRIPT_DIR}/sft_readme_config.sh"

# Stage 2 is not a repeat of Stage 1. It trains the trajectory expert path and
# requires a Stage 1 checkpoint as input. Run this only after Stage 1 has been
# reviewed and accepted as a narrow smoke result.
run_in_tmux_by_default "sft_stage2_nav" "${SCRIPT_DIR}/sft_02_stage2_nav_smoke.sh" "$@"
configure_gpu_selection "$@"

activate_venv

STAGE1_CKPT="${STAGE1_CKPT:-$(latest_checkpoint "${STAGE1_OUTPUT_DIR}")}"

require_dir "${PAI_DIR}" "PAI dataset dir"
require_file "${NAV_ANNOTATIONS}" "nav annotations JSON"
require_dir "${CKPT_DIR_A1}" "A1-format checkpoint dir"
require_file "${CKPT_DIR_A1}/config.json" "A1-format config.json"
require_dir "${STAGE1_CKPT}" "Stage 1 checkpoint dir"
require_file "${STAGE1_CKPT}/model.safetensors.index.json" "Stage 1 model index"

if [[ -e "${STAGE2_OUTPUT_DIR}" && -z "$(latest_checkpoint "${STAGE2_OUTPUT_DIR}")" ]]; then
  log "Stage 2 output dir already exists without a checkpoint: ${STAGE2_OUTPUT_DIR}"
  log "set STAGE2_OUTPUT_DIR to a fresh path before running"
  exit 2
fi
if [[ -n "$(latest_checkpoint "${STAGE2_OUTPUT_DIR}")" ]]; then
  log "Stage 2 output dir already contains a checkpoint: ${STAGE2_OUTPUT_DIR}"
  log "set STAGE2_OUTPUT_DIR to a fresh path before running"
  exit 2
fi
mkdir -p "${STAGE2_OUTPUT_DIR}"

log "Stage 2 nav smoke"
log "purpose=bounded smoke for trajectory expert training after Stage 1"
log "config=sft_stage2_nav"
log "base checkpoint=${CKPT_DIR_A1}"
log "stage1 checkpoint=${STAGE1_CKPT}"
log "deepspeed=disabled by sft_stage2_nav"
log "PAI=${PAI_DIR}"
log "annotations=${NAV_ANNOTATIONS}"
log "output=${STAGE2_OUTPUT_DIR}"
log "selected_gpus=${SFT_GPU_IDS}"
log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
log "nproc_per_node=${NPROC_PER_NODE}"
log "max_steps=${STAGE2_MAX_STEPS:-README epoch-driven run}"
log "W&B=disabled, trainer.report_to=none"

confirm_exact "RUN_STAGE2" "This starts bounded Stage 2 nav SFT."

cd "${RECIPE_DIR}"

overrides=(
  --config-path pkg://alpamayo1_5_sft/configs
  --config-name sft_stage2_nav
  "model.pretrained_model_name_or_path=${CKPT_DIR_A1}"
  "model.stage1_vlm_checkpoint_path=${STAGE1_CKPT}"
  "data.train_dataset.local_dir=${PAI_DIR}"
  "data.train_dataset.annotations_path=${NAV_ANNOTATIONS}"
  "data.val_dataset.local_dir=${PAI_DIR}"
  "data.val_dataset.annotations_path=${NAV_ANNOTATIONS}"
  "trainer.report_to=none"
  "trainer.num_train_epochs=${STAGE2_NUM_TRAIN_EPOCHS}"
  "trainer.save_steps=${STAGE2_SAVE_STEPS}"
  "trainer.save_total_limit=1"
  "paths.output_dir=${STAGE2_OUTPUT_DIR}"
)

if [[ -n "${STAGE2_MAX_STEPS}" ]]; then
  # Same Hydra rule as Stage 1: max_steps is a TrainingArguments field but not
  # predeclared in the YAML config, so append it explicitly.
  overrides+=("+trainer.max_steps=${STAGE2_MAX_STEPS}")
fi

HYDRA_FULL_ERROR=1 WANDB_MODE=disabled CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  torchrun --nproc_per_node "${NPROC_PER_NODE}" \
  -m alpamayo1_5_sft.train_hf \
  "${overrides[@]}"

log "Stage 2 nav smoke finished"
log "latest checkpoint: $(latest_checkpoint "${STAGE2_OUTPUT_DIR}")"
log "next optional eval: scripts/sft_03_eval_stage2_nav.sh"
