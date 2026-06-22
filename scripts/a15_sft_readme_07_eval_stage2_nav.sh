#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/a15_sft_readme_config.sh
source "${SCRIPT_DIR}/a15_sft_readme_config.sh"

run_in_tmux_by_default "a15_sft_07_eval_stage2" "${SCRIPT_DIR}/a15_sft_readme_07_eval_stage2_nav.sh"

activate_venv

STAGE2_CKPT="${STAGE2_CKPT:-$(latest_checkpoint "${STAGE2_OUTPUT_DIR}")}"

require_dir "${PAI_DIR}" "PAI dataset dir"
require_file "${NAV_ANNOTATIONS}" "nav annotations JSON"
require_dir "${STAGE2_CKPT}" "Stage 2 checkpoint dir"

log "Stage 2 evaluation will run."
log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
log "nproc_per_node=${NPROC_PER_NODE}"
log "stage2 checkpoint=${STAGE2_CKPT}"
log "PAI=${PAI_DIR}"
log "annotations=${NAV_ANNOTATIONS}"
log "max_eval_steps=${EVAL_MAX_STEPS}"

confirm_exact "RUN_EVAL" "This step starts evaluation for the Stage 2 checkpoint."

cd "${RECIPE_DIR}"

HYDRA_FULL_ERROR=1 WANDB_MODE=disabled CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  torchrun --nproc_per_node "${NPROC_PER_NODE}" \
  -m alpamayo1_5_sft.evaluate_hf \
  --config-path pkg://alpamayo1_5_sft/configs \
  --config-name sft_stage2_nav \
  "evaluate.eval_ckpt=${STAGE2_CKPT}" \
  "evaluate.max_eval_steps=${EVAL_MAX_STEPS}" \
  "data.val_dataset.local_dir=${PAI_DIR}" \
  "data.val_dataset.annotations_path=${NAV_ANNOTATIONS}" \
  "trainer.report_to=none"

log "Stage 2 evaluation finished"
