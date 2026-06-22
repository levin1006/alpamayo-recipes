#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/a15_sft_readme_config.sh
source "${SCRIPT_DIR}/a15_sft_readme_config.sh"

run_in_tmux_by_default "a15_sft_00_setup_env" "${SCRIPT_DIR}/a15_sft_readme_00_setup_env.sh"

mkdir -p "${ARTIFACT_ROOT}"

log "repo root: ${REPO_ROOT}"
log "recipe dir: ${RECIPE_DIR}"
log "venv dir: ${VENV_DIR}"
log "artifact root: ${ARTIFACT_ROOT}"

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

log "verifying imports"
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

log "environment ready"
log "next: scripts/a15_sft_readme_01_download_pai_nav_chunks.sh"
