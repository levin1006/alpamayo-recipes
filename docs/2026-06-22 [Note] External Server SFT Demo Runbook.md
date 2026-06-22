---
doc_type: note
status: active
created_at: 2026-06-22 15:38:20
updated_at: 2026-06-22 17:24:58
---

# External Server SFT Demo Runbook

## Purpose

Run Track C PAI nav Stage 1 smoke from a disposable Docker container on a shared
external GPU server. The host should keep only persistent data under `/data`; repo
checkout, git metadata, virtualenv, shell history, and temporary credentials stay
inside the container.

This runbook intentionally separates the flow:

1. create a basic container with `/data` mounted;
2. enter the container and clone or pull the personal fork inside `/workspace`;
3. run the repo-provided container preparation script;
4. prepare `/data/datasets/physical_ai_av`;
5. run SFT setup and Stage 1 manually while watching tmux logs.

## Host Layout

- PAI dataset root: `/data/datasets/physical_ai_av`
- SFT artifacts: `/data/alpamayo_sft_artifacts`
- Container repo checkout: `/workspace/alpamayo-recipes`
- Repo URL: `https://github.com/levin1006/alpamayo-recipes.git`

Do not mount host `$HOME`, `.ssh`, `.gitconfig`, Hugging Face cache, or an
existing workspace checkout. Mount only `/data`.

## 1. Create Container

This creates a long-running base container. It does not clone the repo, install
the recipe environment, download data, or start training. The `sleep infinity`
command keeps the container alive after you exit its shell.

```bash
docker run -dit \
  --name alpamayo-sft-demo \
  --gpus '"device=0"' \
  --ipc=host \
  --shm-size=128g \
  -v /data:/data \
  nvcr.io/nvidia/pytorch:25.02-py3 \
  sleep infinity
```

Enter the container:

```bash
docker exec -it alpamayo-sft-demo bash
```

The default exposes only host GPU `0`. To tune the allocation, edit the Docker
device list to match the server, for example `--gpus '"device=<gpu_ids>"'`.
When Docker exposes a subset, the visible GPU IDs inside the container are
usually renumbered from `0`.

## 2. Clone Or Pull Repo First

Run inside the container. Only install `git` first if the base image does not
already include it:

```bash
set -euo pipefail

export HOME=/tmp/sft_container_home
export GIT_TERMINAL_PROMPT=0
export DEBIAN_FRONTEND=noninteractive

if ! command -v git >/dev/null 2>&1; then
  apt-get update
  apt-get install -y --no-install-recommends git ca-certificates
  rm -rf /var/lib/apt/lists/*
fi

mkdir -p /workspace
cd /workspace

if [ -d alpamayo-recipes/.git ]; then
  cd alpamayo-recipes
  git -c credential.helper= fetch origin
  git -c credential.helper= checkout main
  git -c credential.helper= pull --ff-only origin main
else
  git -c credential.helper= clone \
    --branch main \
    https://github.com/levin1006/alpamayo-recipes.git \
    alpamayo-recipes
  cd alpamayo-recipes
fi
```

If you need a specific commit after this runbook is committed and pushed:

```bash
git -c credential.helper= checkout <commit-sha>
```

Avoid embedding tokens in the clone URL. They can be written to `.git/config`.
For this workflow, prefer a public fork branch or a read-only accessible commit.

## 3. Prepare Container Basics From Repo Script

After cloning, use the repo-provided script for the rest of the container-local
environment setup:

```bash
cd /workspace/alpamayo-recipes

scripts/sft_demo_prepare_container_env.sh
```

The script installs or verifies `git`, `curl`, `ca-certificates`, `tmux`,
installs `uv`, links it as `/usr/local/bin/uv`, and creates:

- `/data/datasets/physical_ai_av`
- `/data/alpamayo_sft_artifacts`

`HOME=/tmp/sft_container_home` keeps git/uv state inside the disposable
container. Do not put private tokens into `docker run -e`; Docker users with
inspect privileges may be able to read container environment variables.

## 4. Prepare Data Only

This step prepares only the nav annotation JSON and required PAI 19-chunk sample
payload under `/data/datasets/physical_ai_av`. It is safe to repeat: existing
files are checked first, and missing chunks are downloaded only after explicit
`DOWNLOAD_PAI` confirmation.

```bash
cd /workspace/alpamayo-recipes
scripts/sft_demo_00_prepare_data.sh
```

This command starts a tmux session by default. Attach with:

```bash
tmux attach -t sft_demo_setup
```

If Hugging Face authentication is required, set the token interactively inside
the running container before this step:

```bash
read -rsp "HF_TOKEN: " HF_TOKEN
echo
export HF_TOKEN
```

Expected data-only responsibilities:

- create/sync recipe venv for downloader/status dependencies;
- download and validate `nav_demo_samples.json`;
- create `/data/datasets/physical_ai_av` if missing;
- check existing PAI nav chunks and skip if complete;
- download missing required chunks after `DOWNLOAD_PAI`.

## 5. Prepare Checkpoint

Run full setup when checkpoint preparation is also needed:

```bash
cd /workspace/alpamayo-recipes
scripts/sft_demo_00_setup.sh
```

This command also uses the `sft_demo_setup` tmux session by default.

Expected full setup responsibilities:

- ensure the data-only responsibilities above are satisfied;
- download or reuse Alpamayo 1.5 raw checkpoint;
- convert and verify A1-format checkpoint.

## 6. Run Stage 1

The default uses one visible GPU:

```bash
cd /workspace/alpamayo-recipes
scripts/sft_demo_01_stage1_nav_smoke.sh
```

This command starts `tmux` session `sft_demo_stage1_nav` by default. Attach with:

```bash
tmux attach -t sft_demo_stage1_nav
```

Default execution values are defined inside `scripts/sft_demo_config.sh`:

- `PAI_DIR=/data/datasets/physical_ai_av`
- `ARTIFACT_ROOT=/data/alpamayo_sft_artifacts`
- `SFT_DEFAULT_GPU_IDS=0`
- timestamped Stage 1 output under `/data/alpamayo_sft_artifacts`

Tune these defaults in the script before running if the container-visible GPU
set or data paths differ.

## Security Notes

- Host persistence should be limited to `/data`.
- Git clone/pull happens inside the container, so deleting the container removes
  repo checkout and git metadata.
- Do not mount host `$HOME`, `.ssh`, `.gitconfig`, or HF cache.
- Do not pass private tokens with `docker run -e`.
- Do not put tokens in `REPO_URL`.
- If credentials are needed, enter them interactively inside the running
  container and delete the container after use.

## Non-Claims

This runbook supports a bounded Stage 1 overfit smoke only. It does not establish
a2z readiness, broad PAI nav readiness, generalization, or autonomous-driving
quality/safety improvement.
