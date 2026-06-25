#!/usr/bin/env bash
set -euo pipefail

# Reproduce the meaningful Stage2 continuous-action experiment:
#
#   stage1-sft-frozen + stage2 sft
#
# This is not a bit-exact replay guarantee. It preserves the training contract
# that produced the 2026-06-25 comparison artifact, so a future rerun can check
# whether the same experiment family still lands near the recorded metrics.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RECIPE_DIR="${REPO_ROOT}/recipes/alpamayo1_5_sft"

ARTIFACT_ROOT="${ARTIFACT_ROOT:-/data/alpamayo_sft_artifacts}"
DATASET_DIR="${DATASET_DIR:-/data/datasets/physical_ai_av}"
BASE_A1="${BASE_A1:-${ARTIFACT_ROOT}/Alpamayo-1.5-10B-A1-format}"
STAGE1_CKPT="${STAGE1_CKPT:-${ARTIFACT_ROOT}/output_stage1_nav_smoke_stage1overfit300_20260623_104948/checkpoint-300}"
NAV_SAMPLES="${NAV_SAMPLES:-${ARTIFACT_ROOT}/nav_demo_samples.json}"

RUN_ID="${SFT_RUN_ID:-stage1_sft_frozen_stage2_sft_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ARTIFACT_ROOT}/output_${RUN_ID}}"
LOG_DIR="${SFT_RUN_LOG_DIR:-${REPO_ROOT}/logs/sft_runs}"
LOG_FILE="${LOG_DIR}/${RUN_ID}.log"
DONE_FILE="${LOG_DIR}/${RUN_ID}.done"
FAILED_FILE="${LOG_DIR}/${RUN_ID}.failed"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4}"
NPROC_PER_NODE="${NPROC_PER_NODE:-3}"
MAX_STEPS="${MAX_STEPS:-300}"
WARMUP_STEPS="${WARMUP_STEPS:-10}"
SAVE_STEPS="${SAVE_STEPS:-100}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "${path}" ]]; then
    printf '[sft_experiment] missing %s: %s\n' "${label}" "${path}" >&2
    exit 2
  fi
}

require_dir() {
  local path="$1"
  local label="$2"
  if [[ ! -d "${path}" ]]; then
    printf '[sft_experiment] missing %s: %s\n' "${label}" "${path}" >&2
    exit 2
  fi
}

require_dir "${RECIPE_DIR}" "recipe dir"
require_file "${RECIPE_DIR}/.venv/bin/activate" "recipe venv"
require_dir "${BASE_A1}" "A1-format base checkpoint"
require_file "${BASE_A1}/config.json" "A1-format config"
require_dir "${STAGE1_CKPT}" "Stage1 SFT checkpoint"
require_file "${STAGE1_CKPT}/model.safetensors.index.json" "Stage1 SFT model index"
require_dir "${DATASET_DIR}" "PAI dataset"
require_file "${NAV_SAMPLES}" "nav demo annotations"

if [[ -e "${OUTPUT_ROOT}" && -n "$(find "${OUTPUT_ROOT}" -maxdepth 1 -type d -name 'checkpoint-*' -print -quit 2>/dev/null)" ]]; then
  printf '[sft_experiment] output already contains a checkpoint: %s\n' "${OUTPUT_ROOT}" >&2
  printf '[sft_experiment] choose a fresh SFT_RUN_ID or OUTPUT_ROOT.\n' >&2
  exit 2
fi

mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}"
rm -f "${DONE_FILE}" "${FAILED_FILE}"

exec > >(tee -a "${LOG_FILE}") 2>&1

mark_exit() {
  local status="$1"
  printf '[sft_experiment] finished_at=%s\n' "$(date -Is)"
  printf '[sft_experiment] exit_status=%s\n' "${status}"
  if [[ "${status}" -eq 0 ]]; then
    touch "${DONE_FILE}"
    rm -f "${FAILED_FILE}"
    printf '[sft_experiment] done_marker=%s\n' "${DONE_FILE}"
  else
    touch "${FAILED_FILE}"
    rm -f "${DONE_FILE}"
    printf '[sft_experiment] failed_marker=%s\n' "${FAILED_FILE}"
  fi
}

trap 'status=$?; mark_exit "${status}"; exit "${status}"' EXIT

cd "${RECIPE_DIR}"
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES
export HYDRA_FULL_ERROR=1
export WANDB_MODE=disabled

printf '[sft_experiment] experiment=stage1-sft-frozen + stage2 sft\n'
printf '[sft_experiment] run_id=%s\n' "${RUN_ID}"
printf '[sft_experiment] output=%s\n' "${OUTPUT_ROOT}"
printf '[sft_experiment] log_file=%s\n' "${LOG_FILE}"
printf '[sft_experiment] base_a1=%s\n' "${BASE_A1}"
printf '[sft_experiment] stage1_checkpoint=%s\n' "${STAGE1_CKPT}"
printf '[sft_experiment] dataset=%s\n' "${DATASET_DIR}"
printf '[sft_experiment] annotations=%s\n' "${NAV_SAMPLES}"
printf '[sft_experiment] cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES}"
printf '[sft_experiment] nproc_per_node=%s\n' "${NPROC_PER_NODE}"
printf '[sft_experiment] max_steps=%s\n' "${MAX_STEPS}"
printf '[sft_experiment] tensorboard=%s/tensorboard\n' "${OUTPUT_ROOT}"
printf '[sft_experiment] expected_loader_log=stripped_vlm_prefix=750, missing=0, unexpected=0\n'

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
  "trainer.logging_steps=${LOGGING_STEPS}" \
  "+trainer.max_steps=${MAX_STEPS}" \
  "trainer.warmup_steps=${WARMUP_STEPS}" \
  "trainer.save_steps=${SAVE_STEPS}" \
  "trainer.save_total_limit=${SAVE_TOTAL_LIMIT}" \
  "paths.output_dir=${OUTPUT_ROOT}" \
  "run_name=${RUN_ID}"

printf '[sft_experiment] latest_checkpoint=%s\n' \
  "$(find "${OUTPUT_ROOT}" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null | sort -V | tail -1)"
