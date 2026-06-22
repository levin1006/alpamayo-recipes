#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/a15_sft_readme_config.sh
source "${SCRIPT_DIR}/a15_sft_readme_config.sh"

run_in_tmux_by_default "a15_sft_02_download_ckpt" "${SCRIPT_DIR}/a15_sft_readme_02_download_checkpoint.sh"

activate_venv

log "raw checkpoint target: ${CKPT_DIR_RAW}"
log "This downloads nvidia/Alpamayo-1.5-10B if it is not already present."

if [[ -f "${CKPT_DIR_RAW}/config.json" ]]; then
  log "raw checkpoint already appears present; skipping download"
  log "next: scripts/a15_sft_readme_03_convert_checkpoint_to_a1.sh"
  exit 0
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  log "HF_TOKEN is not set; continuing only works if Hugging Face auth is already cached"
fi

confirm_exact "DOWNLOAD_CKPT" "This step may download about 21GB of model files."

mkdir -p "${CKPT_DIR_RAW}"

if command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download nvidia/Alpamayo-1.5-10B --local-dir "${CKPT_DIR_RAW}"
elif command -v hf >/dev/null 2>&1; then
  hf download nvidia/Alpamayo-1.5-10B --local-dir "${CKPT_DIR_RAW}"
else
  log "neither huggingface-cli nor hf was found"
  exit 127
fi

log "checkpoint download step finished"
log "next: scripts/a15_sft_readme_03_convert_checkpoint_to_a1.sh"
