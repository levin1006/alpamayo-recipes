#!/usr/bin/env bash
# Shared settings for the Alpamayo 1.5 SFT README step scripts.
#
# Edit the variables below, override them from the shell before running a step,
# or pass --gpus to the training/eval scripts.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RECIPE_DIR="${REPO_ROOT}/recipes/alpamayo1_5_sft"
VENV_DIR="${RECIPE_DIR}/.venv"
DEEPSPEED_CONFIG="${RECIPE_DIR}/configs/deepspeed/zero2.json"

PAI_CHUNK_IDS="${PAI_CHUNK_IDS:-214 224 276 317 420 727 728 968 982 1519 1657 1984 2277 2368 2372 2447 2599 2634 2868}"
PAI_DIR="${PAI_DIR:-/mnt/zfs_pool/physical_ai_av}"

if [[ -d /mnt/zfs_pool ]]; then
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

STAGE1_OUTPUT_DIR="${STAGE1_OUTPUT_DIR:-${ARTIFACT_ROOT}/output_stage1_nav_smoke}"
STAGE2_OUTPUT_DIR="${STAGE2_OUTPUT_DIR:-${ARTIFACT_ROOT}/output_stage2_nav_smoke}"

# Keep the first training run bounded by default. Set STAGE1_MAX_STEPS="" to use
# the README's epoch-driven overfit run.
STAGE1_MAX_STEPS="${STAGE1_MAX_STEPS:-20}"
STAGE1_NUM_TRAIN_EPOCHS="${STAGE1_NUM_TRAIN_EPOCHS:-1}"
STAGE1_SAVE_STEPS="${STAGE1_SAVE_STEPS:-20}"

STAGE2_MAX_STEPS="${STAGE2_MAX_STEPS:-20}"
STAGE2_NUM_TRAIN_EPOCHS="${STAGE2_NUM_TRAIN_EPOCHS:-1}"
STAGE2_SAVE_STEPS="${STAGE2_SAVE_STEPS:-20}"

EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-5}"

log() {
  printf '[sft_readme] %s\n' "$*"
}

run_in_tmux_by_default() {
  local session_name="$1"
  local script_path="$2"
  shift 2

  if [[ -n "${TMUX:-}" || "${SFT_DISABLE_TMUX:-no}" == "yes" ]]; then
    return
  fi

  if ! command -v tmux >/dev/null 2>&1; then
    log "tmux is required for the default execution path but was not found on PATH"
    log "install tmux, or run with SFT_DISABLE_TMUX=yes to execute in the current shell"
    exit 127
  fi

  log "starting inside tmux session: ${session_name}"
  log "attach later with: tmux attach -t ${session_name}"
  local command
  printf -v command '%q ' "${script_path}" "$@"
  exec tmux new-session -A -s "${session_name}" "${command}"
}

usage_gpu_args() {
  cat <<'USAGE'
GPU selection:
  --gpus 2          Use one visible GPU, physical GPU 2.
  --gpus 2,3,4      Use physical GPUs 2, 3, and 4.

Environment equivalents:
  SFT_GPU_IDS=2,3,4
  CUDA_VISIBLE_DEVICES=2,3,4

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
          log "--gpus requires a comma-separated GPU list, for example: --gpus 2,3,4"
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
    SFT_GPU_IDS="${CUDA_VISIBLE_DEVICES:-0}"
  fi

  SFT_GPU_IDS="${SFT_GPU_IDS//[[:space:]]/}"
  if [[ ! "${SFT_GPU_IDS}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    log "invalid GPU list: ${SFT_GPU_IDS}"
    log "use a comma-separated list of GPU indexes, for example: --gpus 2,3,4"
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
    log "run scripts/sft_readme_00_setup_env.sh first"
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
  find "${output_dir}" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null | sort -V | tail -1
}
