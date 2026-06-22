#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/a15_sft_readme_config.sh
source "${SCRIPT_DIR}/a15_sft_readme_config.sh"

run_in_tmux_by_default "a15_sft_01_download_pai" "${SCRIPT_DIR}/a15_sft_readme_01_download_pai_nav_chunks.sh"

activate_venv

log "PAI output dir: ${PAI_DIR}"
log "PAI chunk ids: ${PAI_CHUNK_IDS}"
log "This downloads the README nav-demo chunks and can take significant time/storage."

if [[ -z "${HF_TOKEN:-}" ]]; then
  log "HF_TOKEN is not set; continuing only works if Hugging Face auth is already cached"
fi

confirm_exact "DOWNLOAD_PAI" "This step may download PAI dataset chunks."

mkdir -p "${PAI_DIR}"
cd "${REPO_ROOT}"

python scripts/download_pai.py \
  --chunk-ids "${PAI_CHUNK_IDS}" \
  --camera camera_front_wide_120fov camera_cross_left_120fov camera_cross_right_120fov camera_front_tele_30fov \
  --calibration camera_intrinsics sensor_extrinsics \
  --labels egomotion \
  --output-dir "${PAI_DIR}"

log "PAI nav chunks download step finished"
log "next: scripts/a15_sft_readme_02_download_checkpoint.sh"
