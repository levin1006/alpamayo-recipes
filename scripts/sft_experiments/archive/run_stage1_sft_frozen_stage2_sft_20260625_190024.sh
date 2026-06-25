#!/usr/bin/env bash
set -euo pipefail

# Historical archive of the ad-hoc launcher used for:
#
#   stage1_sft_frozen_stage2_sft_20260625_190024
#
# This file is preserved as experiment evidence. It was reconstructed from the
# H100 run log and the local terminal transcript after the ignored logs launcher
# was deleted. Prefer ../run_stage1_sft_frozen_stage2_sft.sh for new reruns.

REPO_ROOT="/workspace/alpamayo-recipes"
RECIPE_DIR="${REPO_ROOT}/recipes/alpamayo1_5_sft"
ARTIFACT_ROOT="/data/alpamayo_sft_artifacts"
RUN_ID="${SFT_RUN_ID:?SFT_RUN_ID is required}"
OUTPUT_ROOT="${ARTIFACT_ROOT}/output_${RUN_ID}"
LOG_DIR="${REPO_ROOT}/logs/sft_runs"
LOG_FILE="${LOG_DIR}/${RUN_ID}.log"
DONE_FILE="${LOG_DIR}/${RUN_ID}.done"
FAILED_FILE="${LOG_DIR}/${RUN_ID}.failed"

mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}"
rm -f "${DONE_FILE}" "${FAILED_FILE}"

exec > >(tee -a "${LOG_FILE}") 2>&1

mark_exit() {
  local status="$1"
  printf '[sft_demo] finished_at=%s\n' "$(date -Is)"
  printf '[sft_demo] exit_status=%s\n' "${status}"
  if [[ "${status}" -eq 0 ]]; then
    touch "${DONE_FILE}"
    rm -f "${FAILED_FILE}"
    printf '[sft_demo] done_marker=%s\n' "${DONE_FILE}"
  else
    touch "${FAILED_FILE}"
    rm -f "${DONE_FILE}"
    printf '[sft_demo] failed_marker=%s\n' "${FAILED_FILE}"
  fi
}

trap 'status=$?; mark_exit "${status}"; exit "${status}"' EXIT

cd "${RECIPE_DIR}"
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4}"
export WANDB_MODE=disabled
export HYDRA_FULL_ERROR=1

NPROC_PER_NODE="${NPROC_PER_NODE:-3}"
BASE_A1="${ARTIFACT_ROOT}/Alpamayo-1.5-10B-A1-format"
STAGE1_CKPT="${ARTIFACT_ROOT}/output_stage1_nav_smoke_stage1overfit300_20260623_104948/checkpoint-300"
DATASET_DIR="/data/datasets/physical_ai_av"
NAV_SAMPLES="${ARTIFACT_ROOT}/nav_demo_samples.json"

printf '[sft_demo] session=stage1_frozen_stage2_sft\n'
printf '[sft_demo] run_id=%s\n' "${RUN_ID}"
printf '[sft_demo] output=%s\n' "${OUTPUT_ROOT}"
printf '[sft_demo] log_file=%s\n' "${LOG_FILE}"
printf '[sft_demo] base_a1=%s\n' "${BASE_A1}"
printf '[sft_demo] stage1=%s\n' "${STAGE1_CKPT}"
printf '[sft_demo] cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES}"
printf '[sft_demo] nproc_per_node=%s\n' "${NPROC_PER_NODE}"
printf '[sft_demo] max_steps=300\n'
printf '[sft_demo] report_to=tensorboard\n'
printf '[sft_demo] tensorboard=%s/tensorboard\n' "${OUTPUT_ROOT}"

torchrun --nproc_per_node "${NPROC_PER_NODE}" \
  -m alpamayo1_5_sft.train_hf \
  --config-path pkg://alpamayo1_5_sft/configs \
  --config-name sft_stage2_nav \
  "model.pretrained_model_name_or_path=${BASE_A1}" \
  "model.stage1_vlm_checkpoint_path=${STAGE1_CKPT}" \
  "data.train_dataset.local_dir=${DATASET_DIR}" \
  "data.train_dataset.annotations_path=${NAV_SAMPLES}" \
  "data.val_dataset.local_dir=${DATASET_DIR}" \
  "data.val_dataset.annotations_path=${NAV_SAMPLES}" \
  "trainer.report_to=tensorboard" \
  "+trainer.logging_dir=${OUTPUT_ROOT}/tensorboard" \
  "trainer.logging_steps=1" \
  "+trainer.max_steps=300" \
  "trainer.warmup_steps=10" \
  "trainer.save_steps=100" \
  "trainer.save_total_limit=3" \
  "paths.output_dir=${OUTPUT_ROOT}"
