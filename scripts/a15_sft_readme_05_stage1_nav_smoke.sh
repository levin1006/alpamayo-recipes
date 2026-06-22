#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/a15_sft_readme_config.sh
source "${SCRIPT_DIR}/a15_sft_readme_config.sh"

run_in_tmux_by_default "a15_sft_05_stage1_nav" "${SCRIPT_DIR}/a15_sft_readme_05_stage1_nav_smoke.sh"

activate_venv

require_dir "${PAI_DIR}" "PAI dataset dir"
require_file "${NAV_ANNOTATIONS}" "nav annotations JSON"
require_dir "${CKPT_DIR_A1}" "A1-format checkpoint dir"
require_file "${CKPT_DIR_A1}/config.json" "A1-format config.json"
require_file "${DEEPSPEED_CONFIG}" "DeepSpeed config"

mkdir -p "${STAGE1_OUTPUT_DIR}"

log "Stage 1 nav smoke will start training."
log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
log "nproc_per_node=${NPROC_PER_NODE}"
log "checkpoint=${CKPT_DIR_A1}"
log "PAI=${PAI_DIR}"
log "annotations=${NAV_ANNOTATIONS}"
log "output=${STAGE1_OUTPUT_DIR}"
log "max_steps=${STAGE1_MAX_STEPS:-README epoch-driven run}"

confirm_exact "RUN_STAGE1" "This step starts bounded Stage 1 nav SFT."

cd "${RECIPE_DIR}"

overrides=(
  --config-path pkg://alpamayo1_5_sft/configs
  --config-name sft_stage1_nav
  "model.checkpoint_path=${CKPT_DIR_A1}"
  "data.train_dataset.local_dir=${PAI_DIR}"
  "data.train_dataset.annotations_path=${NAV_ANNOTATIONS}"
  "data.val_dataset.local_dir=${PAI_DIR}"
  "data.val_dataset.annotations_path=${NAV_ANNOTATIONS}"
  "trainer.deepspeed=${DEEPSPEED_CONFIG}"
  "trainer.report_to=none"
  "trainer.num_train_epochs=${STAGE1_NUM_TRAIN_EPOCHS}"
  "trainer.save_steps=${STAGE1_SAVE_STEPS}"
  "trainer.save_total_limit=1"
  "paths.output_dir=${STAGE1_OUTPUT_DIR}"
)

if [[ -n "${STAGE1_MAX_STEPS}" ]]; then
  overrides+=("trainer.max_steps=${STAGE1_MAX_STEPS}")
fi

HYDRA_FULL_ERROR=1 WANDB_MODE=disabled CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  torchrun --nproc_per_node "${NPROC_PER_NODE}" \
  -m alpamayo1_5_sft.train_hf \
  "${overrides[@]}"

log "Stage 1 nav smoke finished"
log "latest checkpoint: $(latest_checkpoint "${STAGE1_OUTPUT_DIR}")"
log "next: scripts/a15_sft_readme_06_stage2_nav_smoke.sh"
