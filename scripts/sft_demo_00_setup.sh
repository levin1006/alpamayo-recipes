#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/sft_demo_config.sh
source "${SCRIPT_DIR}/sft_demo_config.sh"

# Run this after cloning the repo and preparing the container basics with
# scripts/sft_demo_prepare_container_env.sh. This script owns recipe env sync, PAI nav
# payload readiness, and optional checkpoint preparation; it does not install OS
# packages or clone/pull git repositories.
run_in_tmux_by_default "sft_demo_setup" "${SCRIPT_DIR}/sft_demo_00_setup.sh" "$@"

DATA_ONLY="no"

usage_setup_args() {
  cat <<'USAGE'
Usage:
  scripts/sft_demo_00_setup.sh [--data-only]

Prerequisite:
  Clone/pull the repo first, then run scripts/sft_demo_prepare_container_env.sh.

Options:
  --data-only   Prepare only the nav annotation JSON and PAI nav chunk payload.
                This still syncs the recipe environment because the downloader
                and status checker need Python dependencies.
                The runbook no-argument wrapper is scripts/sft_demo_00_prepare_data.sh.
  -h, --help    Show this help.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-only)
      DATA_ONLY="yes"
      shift
      ;;
    --help|-h)
      usage_setup_args
      exit 0
      ;;
    *)
      log "unknown setup argument: $1"
      usage_setup_args
      exit 2
      ;;
  esac
done

SETUP_LOG_DIR="${SETUP_LOG_DIR:-${REPO_ROOT}/logs/sft_setup}"
PAI_STATUS_PREFIX="${PAI_STATUS_PREFIX:-$(date +%Y%m%d_%H%M%S)_pai_nav_status}"

print_setup_overview() {
  log "Setup will prepare the Stage 1/Stage 2 nav smoke prerequisites."
  log "repo=${REPO_ROOT}"
  log "recipe=${RECIPE_DIR}"
  log "venv=${VENV_DIR}"
  log "PAI_DIR=${PAI_DIR}"
  log "NAV_ANNOTATIONS=${NAV_ANNOTATIONS}"
  log "NAV_ANNOTATIONS_URL=${NAV_ANNOTATIONS_URL}"
  log "CKPT_DIR_RAW=${CKPT_DIR_RAW}"
  log "HF_CACHE_RAW_SNAPSHOT=${HF_CACHE_RAW_SNAPSHOT}"
  log "CKPT_DIR_A1=${CKPT_DIR_A1}"
  log "DATA_ONLY=${DATA_ONLY}"
  log "logs=${SETUP_LOG_DIR}"
}

setup_recipe_environment() {
  log "[1/6] recipe environment"
  mkdir -p "${ARTIFACT_ROOT}" "${SETUP_LOG_DIR}"

  if ! command -v uv >/dev/null 2>&1; then
    log "uv is required but was not found on PATH"
    log "install uv first: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 127
  fi

  cd "${RECIPE_DIR}"

  if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    log "creating recipe-local virtual environment"
    uv venv "${VENV_DIR}" --python 3.12
  else
    log "virtual environment already exists"
  fi

  activate_venv

  log "syncing recipe lockfile"
  uv sync --active

  # Import checks catch missing package wiring before any model load or training
  # begins. These are cheap and safe to repeat.
  log "verifying recipe imports"
  python - <<'PY'
checks = [
    ("alpamayo.data.pai_nav", "PAIDatasetWithNav"),
    ("alpamayo.processor.qwen_processor", "collate_fn_from_model_config"),
    ("alpamayo1_5_sft.train_hf", None),
]

for module_name, attr_name in checks:
    module = __import__(module_name, fromlist=[attr_name] if attr_name else [])
    if attr_name is not None:
        getattr(module, attr_name)
    print(f"ok: {module_name}{'.' + attr_name if attr_name else ''}")
PY
}

ensure_nav_annotations() {
  log "[2/6] nav demo annotations"

  if [[ ! -f "${NAV_ANNOTATIONS}" ]]; then
    log "nav annotations missing: ${NAV_ANNOTATIONS}"
    log "downloading ${NAV_ANNOTATIONS_URL}"
    mkdir -p "$(dirname "${NAV_ANNOTATIONS}")"
    python - "${NAV_ANNOTATIONS_URL}" "${NAV_ANNOTATIONS}" <<'PY'
import sys
from pathlib import Path
from urllib.request import urlopen

url = sys.argv[1]
output = Path(sys.argv[2])

with urlopen(url, timeout=60) as response:
    payload = response.read()

output.write_bytes(payload)
print(f"downloaded {len(payload)} bytes to {output}")
PY
  fi

  python - "${NAV_ANNOTATIONS}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
rows = json.loads(path.read_text())

if not isinstance(rows, list):
    raise SystemExit(f"nav annotations must be a JSON list: {path}")

required = {"clip_id", "t0_relative", "nav_text"}
missing = []
for index, row in enumerate(rows):
    if not isinstance(row, dict):
        missing.append(f"row {index}: not an object")
        continue
    row_missing = sorted(required.difference(row))
    if row_missing:
        missing.append(f"row {index}: missing {row_missing}")

if missing:
    print("nav annotation validation failed:")
    for item in missing[:20]:
        print(f"- {item}")
    if len(missing) > 20:
        print(f"- ... {len(missing) - 20} more")
    raise SystemExit(1)

unique_clips = len({row["clip_id"] for row in rows})
print(f"nav annotations ready: rows={len(rows)}, unique_clips={unique_clips}, path={path}")
PY
}

write_pai_status() {
  cd "${REPO_ROOT}"
  python scripts/check_pai_download_status.py \
    --roots "${PAI_DIR}" \
    --nav-annotations "${NAV_ANNOTATIONS}" \
    --chunk-ids ${PAI_CHUNK_IDS} \
    --output-dir "${SETUP_LOG_DIR}" \
    --prefix "${PAI_STATUS_PREFIX}"
}

has_required_pai_metadata() {
  local missing=()
  local required_paths=(
    "${PAI_DIR}/features.csv"
    "${PAI_DIR}/clip_index.parquet"
    "${PAI_DIR}/metadata/feature_presence.parquet"
  )

  for path in "${required_paths[@]}"; do
    if [[ ! -f "${path}" ]]; then
      missing+=("${path}")
    fi
  done

  if [[ "${#missing[@]}" -gt 0 ]]; then
    log "PAI metadata is not present yet"
    for path in "${missing[@]}"; do
      log "missing metadata: ${path}"
    done
    return 1
  fi

  return 0
}

has_hf_auth() {
  python - <<'PY'
from huggingface_hub import get_token

raise SystemExit(0 if get_token() else 1)
PY
}

ensure_hf_auth_for_download() {
  if [[ -n "${HF_TOKEN:-}" ]]; then
    log "HF_TOKEN is set"
    return
  fi

  if has_hf_auth; then
    log "Hugging Face cached auth token is available"
    return
  fi

  log "Hugging Face auth is required to download the PAI nav demo payload"
  log "Enter an HF token for this container session. Input is hidden."
  local token
  read -rsp "HF_TOKEN: " token
  printf '\n'

  if [[ -z "${token}" ]]; then
    log "HF_TOKEN was empty; cannot download PAI payload"
    exit 2
  fi

  export HF_TOKEN="${token}"
}

assert_pai_status_ready() {
  local summary_json="${SETUP_LOG_DIR}/${PAI_STATUS_PREFIX}__summary.json"

  python - "${summary_json}" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text())
chunk_count = len(summary["chunk_ids_filter"])

errors = []
for item in summary["component_summary"]:
    if item.get("downloaded") != chunk_count:
        errors.append(
            f"{item['component']}: downloaded={item.get('downloaded')} expected={chunk_count}"
        )

joins = summary.get("nav_annotations", {}).get("joins", [])
if not joins:
    errors.append("nav annotation join result is missing")
for join in joins:
    if not join.get("schema_ok"):
        errors.append(f"nav schema missing fields: {join.get('missing_schema')}")
    if join.get("missing_rows") != 0:
        errors.append(f"nav annotation missing_rows={join.get('missing_rows')}")

if errors:
    print("PAI nav payload is not ready:")
    for error in errors:
        print(f"- {error}")
    raise SystemExit(1)

rows = summary["nav_annotations"]["rows"]
clips = summary["nav_annotations"]["unique_clip_count"]
print(f"PAI nav payload ready: rows={rows}, unique_clips={clips}, chunks={chunk_count}")
PY
}

ensure_pai_nav_payload() {
  log "[3/6] PAI nav demo payload"
  require_file "${NAV_ANNOTATIONS}" "nav annotations JSON"

  if [[ -d "${PAI_DIR}" ]] && has_required_pai_metadata && write_pai_status && assert_pai_status_ready; then
    log "PAI nav payload is already ready"
    return
  fi

  log "PAI nav payload is missing or incomplete"
  log "This can download dataset chunks and may take significant time/storage."
  ensure_hf_auth_for_download

  mkdir -p "${PAI_DIR}"
  cd "${REPO_ROOT}"
  python scripts/download_pai.py \
    --chunk-ids "${PAI_CHUNK_IDS}" \
    --camera camera_front_wide_120fov camera_cross_left_120fov camera_cross_right_120fov camera_front_tele_30fov \
    --calibration camera_intrinsics sensor_extrinsics \
    --labels egomotion \
    --output-dir "${PAI_DIR}"

  write_pai_status
  assert_pai_status_ready
}

ensure_raw_checkpoint_input() {
  log "[4/6] raw Alpamayo 1.5 checkpoint input"

  if [[ -f "${CKPT_DIR_RAW}/config.json" ]]; then
    log "raw checkpoint directory exists: ${CKPT_DIR_RAW}"
    return
  fi

  if [[ -f "${HF_CACHE_RAW_SNAPSHOT}/config.json" ]]; then
    log "raw checkpoint target is empty; using HF cache snapshot as conversion input"
    log "cache snapshot=${HF_CACHE_RAW_SNAPSHOT}"
    return
  fi

  log "raw checkpoint is missing from both CKPT_DIR_RAW and HF cache snapshot"
  log "This downloads nvidia/Alpamayo-1.5-10B, about 21GB."
  if [[ -z "${HF_TOKEN:-}" ]]; then
    log "HF_TOKEN is not set; continuing only works if Hugging Face auth is already cached"
  fi

  confirm_exact "DOWNLOAD_CKPT" "Download nvidia/Alpamayo-1.5-10B now?"

  mkdir -p "${CKPT_DIR_RAW}"
  if command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download nvidia/Alpamayo-1.5-10B --local-dir "${CKPT_DIR_RAW}"
  elif command -v hf >/dev/null 2>&1; then
    hf download nvidia/Alpamayo-1.5-10B --local-dir "${CKPT_DIR_RAW}"
  else
    log "neither huggingface-cli nor hf was found"
    exit 127
  fi
}

convert_checkpoint_if_needed() {
  log "[5/6] A1-format checkpoint conversion"

  local raw_input="${CKPT_DIR_RAW}"
  if [[ ! -f "${raw_input}/config.json" && -f "${HF_CACHE_RAW_SNAPSHOT}/config.json" ]]; then
    raw_input="${HF_CACHE_RAW_SNAPSHOT}"
  fi

  require_file "${raw_input}/config.json" "raw checkpoint config.json"

  if [[ -e "${CKPT_DIR_A1}/config.json" && "${OVERWRITE_A1:-no}" != "yes" ]]; then
    log "A1-format checkpoint already exists: ${CKPT_DIR_A1}"
    log "set OVERWRITE_A1=yes to recreate it"
    return
  fi

  log "raw input=${raw_input}"
  log "A1 output=${CKPT_DIR_A1}"
  mkdir -p "$(dirname "${CKPT_DIR_A1}")"

  local cmd=(python scripts/convert_checkpoint.py to-a1 --input "${raw_input}" --output "${CKPT_DIR_A1}")
  if [[ "${OVERWRITE_A1:-no}" == "yes" ]]; then
    cmd+=(--overwrite)
  fi

  cd "${REPO_ROOT}"
  "${cmd[@]}"
}

verify_a1_checkpoint() {
  log "[6/6] A1-format checkpoint verification"
  require_dir "${CKPT_DIR_A1}" "A1-format checkpoint dir"
  require_file "${CKPT_DIR_A1}/config.json" "A1-format config.json"
  require_file "${CKPT_DIR_A1}/model.safetensors.index.json" "A1-format model index"

  python - "${CKPT_DIR_A1}" <<'PY'
import json
import sys
from pathlib import Path

ckpt = Path(sys.argv[1])
config = json.loads((ckpt / "config.json").read_text())

errors = []
if config.get("model_type") != "alpamayo_r1":
    errors.append(f"model_type={config.get('model_type')!r}, expected 'alpamayo_r1'")
if config.get("architectures") != ["AlpamayoR1"]:
    errors.append(f"architectures={config.get('architectures')!r}, expected ['AlpamayoR1']")
if "alpamayo1_5." in json.dumps(config):
    errors.append("config still contains alpamayo1_5.* targets")

weight_shards = sorted(ckpt.glob("model-*-of-*.safetensors"))
if not weight_shards:
    errors.append("no model shard symlinks/files found")

if errors:
    print("A1 checkpoint verification failed:")
    for error in errors:
        print(f"- {error}")
    raise SystemExit(1)

print("A1 checkpoint verification passed")
print(f"checkpoint: {ckpt}")
print(f"weight shards: {len(weight_shards)}")
print(f"model_type: {config.get('model_type')}")
print(f"architectures: {config.get('architectures')}")
PY
}

print_setup_overview
setup_recipe_environment
ensure_nav_annotations
ensure_pai_nav_payload
if [[ "${DATA_ONLY}" == "yes" ]]; then
  log "data-only setup complete"
  exit 0
fi
ensure_raw_checkpoint_input
convert_checkpoint_if_needed
verify_a1_checkpoint

log "setup complete"
log "next Stage 1 smoke: scripts/sft_demo_01_stage1_nav_smoke.sh"
