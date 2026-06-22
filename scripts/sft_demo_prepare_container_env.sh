#!/usr/bin/env bash
set -euo pipefail

# Run this after cloning the repo inside a disposable GPU container.
# It prepares only container-local tooling and persistent /data directories.

CONTAINER_HOME="${CONTAINER_HOME:-/tmp/sft_container_home}"
PAI_DIR="${PAI_DIR:-/data/datasets/physical_ai_av}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/data/alpamayo_sft_artifacts}"

log() {
  printf '[sft_container_prepare] %s\n' "$*"
}

install_os_packages_if_needed() {
  local missing=()
  for command_name in git curl ca-certificates tmux; do
    if ! command -v "${command_name}" >/dev/null 2>&1; then
      missing+=("${command_name}")
    fi
  done

  if [[ "${#missing[@]}" -eq 0 ]]; then
    return
  fi

  if ! command -v apt-get >/dev/null 2>&1; then
    log "missing commands: ${missing[*]}"
    log "apt-get is unavailable; install them manually before continuing"
    exit 127
  fi

  log "installing base packages: ${missing[*]}"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y --no-install-recommends git curl ca-certificates tmux
  rm -rf /var/lib/apt/lists/*
}

install_uv_if_needed() {
  if command -v uv >/dev/null 2>&1; then
    log "uv already available: $(command -v uv)"
    return
  fi

  log "installing uv under ${CONTAINER_HOME}"
  mkdir -p "${CONTAINER_HOME}"
  HOME="${CONTAINER_HOME}" curl -LsSf https://astral.sh/uv/install.sh | sh

  local uv_path="${CONTAINER_HOME}/.local/bin/uv"
  if [[ ! -x "${uv_path}" ]]; then
    log "uv install did not produce ${uv_path}"
    exit 127
  fi

  if [[ -e /usr/local/bin/uv ]]; then
    log "uv already linked at /usr/local/bin/uv"
  elif [[ -w /usr/local/bin ]]; then
    ln -s "${uv_path}" /usr/local/bin/uv
    log "linked uv to /usr/local/bin/uv"
  else
    log "cannot write /usr/local/bin/uv"
    log "rerun as a user with permission, or manually add ${CONTAINER_HOME}/.local/bin to PATH"
    exit 127
  fi
}

prepare_data_dirs() {
  log "preparing persistent directories"
  mkdir -p "${PAI_DIR}" "${ARTIFACT_ROOT}"
  log "PAI_DIR=${PAI_DIR}"
  log "ARTIFACT_ROOT=${ARTIFACT_ROOT}"
}

print_next_steps() {
  cat <<EOF

Container environment is ready.

Next commands:

  cd /workspace/alpamayo-recipes

  scripts/sft_demo_00_prepare_data.sh

  scripts/sft_demo_00_setup.sh

  scripts/sft_demo_01_stage1_nav_smoke.sh

EOF
}

install_os_packages_if_needed
install_uv_if_needed
prepare_data_dirs
print_next_steps
