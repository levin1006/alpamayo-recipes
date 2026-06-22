#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# No-argument entrypoint for the runbook's data preparation phase.
exec "${SCRIPT_DIR}/sft_demo_00_setup.sh" --data-only
