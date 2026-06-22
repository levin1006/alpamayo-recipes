#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/sft_demo_config.sh
source "${SCRIPT_DIR}/sft_demo_config.sh"

# Evaluation is intentionally separate from SFT. Running it on the same 20-row
# demo payload can only support an overfit/sanity interpretation, not a
# generalization or autonomous-driving quality claim.
run_in_tmux_by_default "sft_demo_eval_stage2_nav" "${SCRIPT_DIR}/sft_demo_03_eval_stage2_nav.sh" "$@"
configure_gpu_selection "$@"

activate_venv

STAGE2_CKPT="${STAGE2_CKPT:-$(latest_checkpoint "${STAGE2_OUTPUT_DIR}")}"

require_dir "${PAI_DIR}" "PAI dataset dir"
require_file "${NAV_ANNOTATIONS}" "nav annotations JSON"
require_dir "${STAGE2_CKPT}" "Stage 2 checkpoint dir"

log "Stage 2 nav evaluation"
log "purpose=same-demo smoke/evidence collection only"
log "config=sft_stage2_nav"
log "stage2 checkpoint=${STAGE2_CKPT}"
log "PAI=${PAI_DIR}"
log "annotations=${NAV_ANNOTATIONS}"
log "selected_gpus=${SFT_GPU_IDS}"
log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
log "nproc_per_node=${NPROC_PER_NODE}"
log "max_eval_steps=${EVAL_MAX_STEPS}"
log "W&B=disabled, trainer.report_to=none"

confirm_exact "RUN_EVAL" "This starts evaluation for the Stage 2 checkpoint."

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
