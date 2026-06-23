#!/usr/bin/env bash
# Shared settings for the Alpamayo 1.5 SFT helper scripts.
#
# Edit the defaults below to tune the runbook. The scripts can still accept
# environment overrides, but the documented path uses no CLI environment setup.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RECIPE_DIR="${REPO_ROOT}/recipes/alpamayo1_5_sft"
VENV_DIR="${RECIPE_DIR}/.venv"
DEEPSPEED_CONFIG="${RECIPE_DIR}/configs/deepspeed/zero2.json"

PAI_CHUNK_IDS="${PAI_CHUNK_IDS:-214 224 276 317 420 727 728 968 982 1519 1657 1984 2277 2368 2372 2447 2599 2634 2868}"
SFT_DEFAULT_GPU_IDS="${SFT_DEFAULT_GPU_IDS:-4}"
SFT_RUN_ID="${SFT_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

if [[ -d /data || -w / ]]; then
  DEFAULT_PAI_DIR="/data/datasets/physical_ai_av"
else
  DEFAULT_PAI_DIR="${HOME}/datasets/physical_ai_av"
fi

PAI_DIR="${PAI_DIR:-${DEFAULT_PAI_DIR}}"

if [[ -d /data || -w / ]]; then
  DEFAULT_ARTIFACT_ROOT="/data/alpamayo_sft_artifacts"
elif [[ -d /mnt/zfs_pool ]]; then
  DEFAULT_ARTIFACT_ROOT="/mnt/zfs_pool/alpamayo_sft_artifacts"
else
  DEFAULT_ARTIFACT_ROOT="${HOME}/alpamayo_sft_artifacts"
fi

ARTIFACT_ROOT="${ARTIFACT_ROOT:-${DEFAULT_ARTIFACT_ROOT}}"
NAV_ANNOTATIONS_URL="${NAV_ANNOTATIONS_URL:-https://raw.githubusercontent.com/NVlabs/alpamayo1.5/main/notebooks/nav_demo_samples.json}"
NAV_ANNOTATIONS="${NAV_ANNOTATIONS:-${ARTIFACT_ROOT}/nav_demo_samples.json}"
CKPT_DIR_RAW="${CKPT_DIR_RAW:-${ARTIFACT_ROOT}/Alpamayo-1.5-10B}"
CKPT_DIR_A1="${CKPT_DIR_A1:-${ARTIFACT_ROOT}/Alpamayo-1.5-10B-A1-format}"

# Existing HF cache snapshot from prior inference/preflight runs. Used only as a
# fallback input for conversion when CKPT_DIR_RAW has not been downloaded yet.
HF_CACHE_RAW_SNAPSHOT="${HF_CACHE_RAW_SNAPSHOT:-${HOME}/.cache/huggingface/hub/models--nvidia--Alpamayo-1.5-10B/snapshots/f11cd25b758ab560114019b555dde2a8b92d88b4}"

NPROC_PER_NODE="${NPROC_PER_NODE:-}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
SFT_GPU_IDS="${SFT_GPU_IDS:-}"

STAGE1_OUTPUT_DIR="${STAGE1_OUTPUT_DIR:-${ARTIFACT_ROOT}/output_stage1_nav_smoke_${SFT_RUN_ID}}"
STAGE2_OUTPUT_DIR="${STAGE2_OUTPUT_DIR:-${ARTIFACT_ROOT}/output_stage2_nav_smoke_${SFT_RUN_ID}}"

# Keep the first training run bounded by default. Set STAGE1_MAX_STEPS="" to use
# the README's epoch-driven overfit run.
STAGE1_MAX_STEPS="${STAGE1_MAX_STEPS:-300}"
STAGE1_NUM_TRAIN_EPOCHS="${STAGE1_NUM_TRAIN_EPOCHS:-1}"
STAGE1_SAVE_STEPS="${STAGE1_SAVE_STEPS:-300}"
STAGE1_LOGGING_STEPS="${STAGE1_LOGGING_STEPS:-1}"
STAGE1_WARMUP_STEPS="${STAGE1_WARMUP_STEPS:-5}"
STAGE1_REPORT_TO="${STAGE1_REPORT_TO:-tensorboard}"
STAGE1_TENSORBOARD_DIR="${STAGE1_TENSORBOARD_DIR:-${STAGE1_OUTPUT_DIR}/tensorboard}"

STAGE2_MAX_STEPS="${STAGE2_MAX_STEPS:-20}"
STAGE2_NUM_TRAIN_EPOCHS="${STAGE2_NUM_TRAIN_EPOCHS:-1}"
STAGE2_SAVE_STEPS="${STAGE2_SAVE_STEPS:-20}"

EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-5}"

SFT_RUN_LOG_DIR="${SFT_RUN_LOG_DIR:-${REPO_ROOT}/logs/sft_runs}"
SFT_TMUX_HOLD="${SFT_TMUX_HOLD:-yes}"

log() {
  printf '[sft_demo] %s\n' "$*"
}

run_in_tmux_by_default() {
  local session_name="$1"
  local script_path="$2"
  shift 2

  if [[ -n "${TMUX:-}" || "${SFT_DISABLE_TMUX:-no}" == "yes" ]]; then
    return
  fi

  for arg in "$@"; do
    if [[ "${arg}" == "--help" || "${arg}" == "-h" ]]; then
      return
    fi
  done

  if ! command -v tmux >/dev/null 2>&1; then
    log "tmux is required for the default execution path but was not found on PATH"
    log "install tmux, or run with SFT_DISABLE_TMUX=yes to execute in the current shell"
    exit 127
  fi

  mkdir -p "${SFT_RUN_LOG_DIR}"

  local log_base="${SFT_RUN_ID}_${session_name}"
  local log_file="${SFT_RUN_LOG_DIR}/${log_base}.log"
  local done_file="${SFT_RUN_LOG_DIR}/${log_base}.done"
  local failed_file="${SFT_RUN_LOG_DIR}/${log_base}.failed"

  log "starting inside tmux session: ${session_name}"
  log "attach later with: tmux attach -t ${session_name}"
  log "log file: ${log_file}"
  log "done marker: ${done_file}"
  log "failed marker: ${failed_file}"

  local runner
  read -r -d '' runner <<'BASH' || true
set -o pipefail

script_path="$1"
shift

mkdir -p "$(dirname "${SFT_TMUX_LOG_FILE}")"
rm -f "${SFT_TMUX_DONE_FILE}" "${SFT_TMUX_FAILED_FILE}"

write_result_marker() {
  local status="$1"

  printf '[sft_demo] finished_at=%s\n' "$(date -Is)"
  printf '[sft_demo] exit_status=%s\n' "${status}"

  if [[ "${status}" -eq 0 ]]; then
    touch "${SFT_TMUX_DONE_FILE}"
    rm -f "${SFT_TMUX_FAILED_FILE}"
    printf '[sft_demo] done_marker=%s\n' "${SFT_TMUX_DONE_FILE}"
  else
    touch "${SFT_TMUX_FAILED_FILE}"
    rm -f "${SFT_TMUX_DONE_FILE}"
    printf '[sft_demo] failed_marker=%s\n' "${SFT_TMUX_FAILED_FILE}"
  fi
}

handle_signal() {
  local status="$1"
  {
    printf '[sft_demo] interrupted_by_signal exit_status=%s\n' "${status}"
    write_result_marker "${status}"
  } 2>&1 | tee -a "${SFT_TMUX_LOG_FILE}"
  exit "${status}"
}

trap 'handle_signal 130' INT
trap 'handle_signal 143' TERM
trap 'handle_signal 129' HUP

{
  printf '[sft_demo] session=%s\n' "${SFT_TMUX_SESSION}"
  printf '[sft_demo] started_at=%s\n' "$(date -Is)"
  printf '[sft_demo] command=%q' "${script_path}"
  for arg in "$@"; do
    printf ' %q' "${arg}"
  done
  printf '\n'
  printf '[sft_demo] log_file=%s\n' "${SFT_TMUX_LOG_FILE}"

  "${script_path}" "$@"
  status=$?

  write_result_marker "${status}"

  exit "${status}"
} 2>&1 | tee -a "${SFT_TMUX_LOG_FILE}"

status="${PIPESTATUS[0]}"
if [[ "${SFT_TMUX_HOLD}" != "no" ]]; then
  printf '\n[sft_demo] session complete; press Enter to close this tmux pane... '
  read -r _
fi
exit "${status}"
BASH

  local command
  printf -v command '%q ' \
    env \
    "SFT_TMUX_SESSION=${session_name}" \
    "SFT_TMUX_LOG_FILE=${log_file}" \
    "SFT_TMUX_DONE_FILE=${done_file}" \
    "SFT_TMUX_FAILED_FILE=${failed_file}" \
    "SFT_TMUX_HOLD=${SFT_TMUX_HOLD}" \
    bash -lc "${runner}" sft_tmux_runner "${script_path}" "$@"

  if [[ ! -t 1 ]]; then
    log "no interactive terminal detected; starting detached tmux session"
    tmux new-session -d -s "${session_name}" "${command}"
    exit 0
  fi

  exec tmux new-session -A -s "${session_name}" "${command}"
}

usage_gpu_args() {
  cat <<'USAGE'
GPU selection:
  --gpus 4          Use the default reserved GPU for this H100 demo run.
  --gpus 0          Use a different single visible GPU.
  --gpus 0,1,2      Use visible GPUs 0, 1, and 2.
  --gpus 0,1,2,3,4  Use all five visible GPUs when available.

Environment equivalents:
  SFT_GPU_IDS=4
  CUDA_VISIBLE_DEVICES=4

NPROC_PER_NODE defaults to the number of selected GPU IDs. Override it only
when intentionally running fewer processes than visible GPUs.
USAGE
}

count_gpu_ids() {
  local ids="$1"
  local compact="${ids//[[:space:]]/}"

  if [[ -z "${compact}" ]]; then
    printf '0\n'
    return
  fi

  local without_commas="${compact//,/}"
  printf '%s\n' "$(( ${#compact} - ${#without_commas} + 1 ))"
}

configure_gpu_selection() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --gpus)
        if [[ $# -lt 2 || -z "$2" ]]; then
          log "--gpus requires a comma-separated GPU list, for example: --gpus 0,1,2"
          exit 2
        fi
        SFT_GPU_IDS="$2"
        shift 2
        ;;
      --gpus=*)
        SFT_GPU_IDS="${1#--gpus=}"
        shift
        ;;
      --help|-h)
        usage_gpu_args
        exit 0
        ;;
      *)
        log "unknown argument: $1"
        usage_gpu_args
        exit 2
        ;;
    esac
  done

  if [[ -z "${SFT_GPU_IDS}" ]]; then
    SFT_GPU_IDS="${CUDA_VISIBLE_DEVICES:-${SFT_DEFAULT_GPU_IDS}}"
  fi

  SFT_GPU_IDS="${SFT_GPU_IDS//[[:space:]]/}"
  if [[ ! "${SFT_GPU_IDS}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    log "invalid GPU list: ${SFT_GPU_IDS}"
    log "use a comma-separated list of GPU indexes, for example: --gpus 0,1,2"
    exit 2
  fi

  CUDA_VISIBLE_DEVICES="${SFT_GPU_IDS}"
  export CUDA_VISIBLE_DEVICES

  if [[ -z "${NPROC_PER_NODE}" ]]; then
    NPROC_PER_NODE="$(count_gpu_ids "${SFT_GPU_IDS}")"
  fi

  if [[ ! "${NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
    log "invalid NPROC_PER_NODE: ${NPROC_PER_NODE}"
    exit 2
  fi
}

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "${path}" ]]; then
    log "missing ${label}: ${path}"
    exit 2
  fi
}

require_dir() {
  local path="$1"
  local label="$2"
  if [[ ! -d "${path}" ]]; then
    log "missing ${label}: ${path}"
    exit 2
  fi
}

activate_venv() {
  if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    log "venv is missing: ${VENV_DIR}"
    log "run scripts/sft_demo_00_setup.sh first"
    exit 2
  fi
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
}

confirm_exact() {
  local expected="$1"
  local prompt="$2"
  printf '%s\n' "${prompt}"
  printf 'Type %s to continue: ' "${expected}"
  local answer
  read -r answer
  if [[ "${answer}" != "${expected}" ]]; then
    log "cancelled"
    exit 130
  fi
}

latest_checkpoint() {
  local output_dir="$1"
  if [[ -z "${output_dir}" || ! -d "${output_dir}" ]]; then
    return 0
  fi
  find "${output_dir}" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null | sort -V | tail -1
}
