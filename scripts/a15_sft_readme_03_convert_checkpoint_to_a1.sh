#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/a15_sft_readme_config.sh
source "${SCRIPT_DIR}/a15_sft_readme_config.sh"

run_in_tmux_by_default "a15_sft_03_convert_a1" "${SCRIPT_DIR}/a15_sft_readme_03_convert_checkpoint_to_a1.sh"

activate_venv

RAW_INPUT="${CKPT_DIR_RAW}"
if [[ ! -f "${RAW_INPUT}/config.json" && -f "${HF_CACHE_RAW_SNAPSHOT}/config.json" ]]; then
  RAW_INPUT="${HF_CACHE_RAW_SNAPSHOT}"
  log "CKPT_DIR_RAW is not populated; using HF cache snapshot as conversion input"
fi

require_file "${RAW_INPUT}/config.json" "raw checkpoint config.json"

if [[ -e "${CKPT_DIR_A1}/config.json" && "${OVERWRITE_A1:-no}" != "yes" ]]; then
  log "A1-format checkpoint already exists: ${CKPT_DIR_A1}"
  log "set OVERWRITE_A1=yes before running this script if you want to recreate it"
  log "next: scripts/a15_sft_readme_04_verify_a1_checkpoint.sh"
  exit 0
fi

log "raw checkpoint input: ${RAW_INPUT}"
log "A1-format output: ${CKPT_DIR_A1}"

mkdir -p "$(dirname "${CKPT_DIR_A1}")"

cmd=(python scripts/convert_checkpoint.py to-a1 --input "${RAW_INPUT}" --output "${CKPT_DIR_A1}")
if [[ "${OVERWRITE_A1:-no}" == "yes" ]]; then
  cmd+=(--overwrite)
fi

cd "${REPO_ROOT}"
"${cmd[@]}"

log "checkpoint conversion step finished"
log "next: scripts/a15_sft_readme_04_verify_a1_checkpoint.sh"
