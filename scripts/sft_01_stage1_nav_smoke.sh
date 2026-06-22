#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/sft_readme_config.sh
source "${SCRIPT_DIR}/sft_readme_config.sh"

# Stage 1 answers the first wiring question:
# Can the base VLM SFT path learn from the tiny 20-row PAI nav demo payload?
#
# This is not a quality or generalization test. It is a bounded overfit smoke:
# if loss cannot move on these same rows, model/dataset/collator/trainer wiring
# is suspect.
run_in_tmux_by_default "sft_stage1_nav" "${SCRIPT_DIR}/sft_01_stage1_nav_smoke.sh" "$@"
configure_gpu_selection "$@"

activate_venv

require_dir "${PAI_DIR}" "PAI dataset dir"
require_file "${NAV_ANNOTATIONS}" "nav annotations JSON"
require_dir "${CKPT_DIR_A1}" "A1-format checkpoint dir"
require_file "${CKPT_DIR_A1}/config.json" "A1-format config.json"
require_file "${DEEPSPEED_CONFIG}" "DeepSpeed config"

# Refuse to silently reuse an old output directory. Reusing training outputs
# makes it hard to tell which checkpoint and loss log came from this smoke.
if [[ -e "${STAGE1_OUTPUT_DIR}" && -z "$(latest_checkpoint "${STAGE1_OUTPUT_DIR}")" ]]; then
  log "Stage 1 output dir already exists without a checkpoint: ${STAGE1_OUTPUT_DIR}"
  log "set STAGE1_OUTPUT_DIR to a fresh path before running"
  exit 2
fi
if [[ -n "$(latest_checkpoint "${STAGE1_OUTPUT_DIR}")" ]]; then
  log "Stage 1 output dir already contains a checkpoint: ${STAGE1_OUTPUT_DIR}"
  log "set STAGE1_OUTPUT_DIR to a fresh path before running"
  exit 2
fi
mkdir -p "${STAGE1_OUTPUT_DIR}"

log "Stage 1 nav smoke"
log "purpose=bounded overfit smoke for base VLM SFT wiring"
log "config=sft_stage1_nav"
log "model checkpoint=${CKPT_DIR_A1}"
log "deepspeed=${DEEPSPEED_CONFIG}"
log "PAI=${PAI_DIR}"
log "annotations=${NAV_ANNOTATIONS}"
log "output=${STAGE1_OUTPUT_DIR}"
log "selected_gpus=${SFT_GPU_IDS}"
log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
log "nproc_per_node=${NPROC_PER_NODE}"
log "max_steps=${STAGE1_MAX_STEPS:-README epoch-driven run}"
log "W&B=disabled, trainer.report_to=none"

confirm_exact "RUN_STAGE1" "This starts bounded Stage 1 nav SFT."

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
  # max_steps is accepted by TrainingArguments, but it is not declared in the
  # Hydra config tree. The leading + appends it intentionally.
  overrides+=("+trainer.max_steps=${STAGE1_MAX_STEPS}")
fi

HYDRA_FULL_ERROR=1 WANDB_MODE=disabled CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  torchrun --nproc_per_node "${NPROC_PER_NODE}" \
  -m alpamayo1_5_sft.train_hf \
  "${overrides[@]}"

log "Stage 1 nav smoke finished"
log "latest checkpoint: $(latest_checkpoint "${STAGE1_OUTPUT_DIR}")"
log "review Stage 1 before running Stage 2"
