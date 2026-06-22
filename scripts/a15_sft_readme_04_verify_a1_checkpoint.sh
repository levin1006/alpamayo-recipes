#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/a15_sft_readme_config.sh
source "${SCRIPT_DIR}/a15_sft_readme_config.sh"

run_in_tmux_by_default "a15_sft_04_verify_a1" "${SCRIPT_DIR}/a15_sft_readme_04_verify_a1_checkpoint.sh"

activate_venv

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

config_text = json.dumps(config)
if "alpamayo1_5." in config_text:
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

log "A1-format checkpoint is ready"
log "next: scripts/a15_sft_readme_05_stage1_nav_smoke.sh"
