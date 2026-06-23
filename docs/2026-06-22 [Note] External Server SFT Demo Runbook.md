---
doc_type: note
status: active
created_at: 2026-06-22 15:38:20
updated_at: 2026-06-23 09:53:30
---

# External Server SFT Demo Runbook

## Purpose

Run Track C PAI nav Stage 1 smoke from a Docker container on a shared external
GPU server. The H100 host owns a normal git workspace under
`/home/aidev1/workspace/alpamayo-recipes`; the container mounts that workspace
at `/workspace/alpamayo-recipes`, uses `/data/datasets` for the PAI dataset,
and writes large checkpoints and training outputs under
`/data/alpamayo_sft_artifacts`.

This runbook intentionally separates the flow:

1. maintain the repo from the H100 host workspace;
2. create a long-running container with GPU `0,1,2,3,4`, `/data`, the host
   workspace, and X11 mounts;
3. run the repo-provided container preparation script;
4. prepare `/data/datasets/physical_ai_av`;
5. run SFT setup and Stage 1 manually while watching tmux logs.

## Host Layout

- PAI dataset root: `/data/datasets/physical_ai_av`
- SFT artifacts: `/data/alpamayo_sft_artifacts`
- H100 host workspace: `/home/aidev1/workspace/alpamayo-recipes`
- Container workspace mount: `/workspace/alpamayo-recipes`
- Repo URL: `https://github.com/levin1006/alpamayo-recipes.git`

Do not keep repo source only inside the container. The container is replaceable;
the H100 host workspace is the server-side git checkout. `/data` remains for
large datasets, checkpoints, and run artifacts.

Keep `/data/datasets` and `/data/alpamayo_sft_artifacts` writable by the host
user that owns the workspace. The validated H100 run uses `aidev1:aidev1` for
those paths, so files created by the non-root container remain editable from the
host.

## 0. Prepare H100 Host Workspace

Run on the H100 host, outside the container:

```bash
mkdir -p /home/aidev1/workspace
cd /home/aidev1/workspace

if [ -d alpamayo-recipes/.git ]; then
  cd alpamayo-recipes
  git fetch origin
  git checkout main
  git pull --ff-only origin main
else
  git clone https://github.com/levin1006/alpamayo-recipes.git alpamayo-recipes
  cd alpamayo-recipes
fi
```

This host checkout may keep git credentials if needed. It is intentionally
separate from the local development workspace:

- local workspace: `/home/user/Workspace/alpamayo-recipes`
- H100 workspace: `/home/aidev1/workspace/alpamayo-recipes`
- container workspace: `/workspace/alpamayo-recipes`

## 1. Create Container

This creates a long-running base container. It does not clone the repo, install
the recipe environment, download data, or start training. The `sleep infinity`
command keeps the container alive after you exit its shell. The H100 server uses
the readable local image tag `alpamayo-sft-h100:25.02`, a local non-root image
built from `nvcr.io/nvidia/pytorch:25.02-py3` with `git`, `curl`, `tmux`, `uv`,
and the `aidev1` UID/GID baked in.

```bash
cat >/tmp/alpamayo-sft-h100.Dockerfile <<'EOF'
FROM nvcr.io/nvidia/pytorch:25.02-py3
ARG UID_NUM=1002
ARG GID_NUM=1002
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends git curl ca-certificates tmux \
 && rm -rf /var/lib/apt/lists/*
RUN if ! getent group "${GID_NUM}" >/dev/null; then groupadd -g "${GID_NUM}" aidev1; fi \
 && if ! id -u "${UID_NUM}" >/dev/null 2>&1; then useradd -m -u "${UID_NUM}" -g "${GID_NUM}" -s /bin/bash aidev1; fi
RUN curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR=/usr/local/bin sh \
 && chmod 0755 /usr/local/bin/uv /usr/local/bin/uvx
ENV HOME=/home/aidev1
WORKDIR /workspace/alpamayo-recipes
USER ${UID_NUM}:${GID_NUM}
EOF

docker build \
  --build-arg UID_NUM="$(id -u)" \
  --build-arg GID_NUM="$(id -g)" \
  -t alpamayo-sft-h100:25.02 \
  -f /tmp/alpamayo-sft-h100.Dockerfile \
  /tmp

X11_ARGS=()
if [ -S /tmp/.X11-unix/X0 ] || [ -d /tmp/.X11-unix ]; then
  X11_ARGS+=(-e DISPLAY="${DISPLAY:-:0}")
  X11_ARGS+=(-e QT_X11_NO_MITSHM=1)
  X11_ARGS+=(-v /tmp/.X11-unix:/tmp/.X11-unix:rw)
fi
if [ -f "${HOME}/.Xauthority" ]; then
  X11_ARGS+=(-e XAUTHORITY=/home/aidev1/.Xauthority)
  X11_ARGS+=(-v "${HOME}/.Xauthority:/home/aidev1/.Xauthority:ro")
fi

docker run -dit \
  --name alpamayo-sft-h100 \
  --gpus '"device=0,1,2,3,4"' \
  --ipc=host \
  --shm-size=128g \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v /data:/data \
  -v /home/aidev1/workspace/alpamayo-recipes:/workspace/alpamayo-recipes \
  "${X11_ARGS[@]}" \
  alpamayo-sft-h100:25.02 \
  sleep infinity
```

Enter the container:

```bash
docker exec -it alpamayo-sft-h100 bash
```

Docker exposes host GPUs `0,1,2,3,4` to the container. The demo scripts default
to GPU `4`, because that is the currently reserved GPU. To run on all exposed
GPUs later, edit `SFT_DEFAULT_GPU_IDS` in `scripts/sft_demo_config.sh` or pass
`--gpus 0,1,2,3,4` to the stage script.

## 3. Prepare Container Basics From Repo Script

Use the repo-provided script for the rest of the container-local environment
setup:

```bash
cd /workspace/alpamayo-recipes

scripts/sft_demo_prepare_container_env.sh
```

The script installs or verifies `git`, `curl`, `ca-certificates`, `tmux`,
installs `uv`, links it as `/usr/local/bin/uv`, and creates:

- `/data/datasets/physical_ai_av`
- `/data/alpamayo_sft_artifacts`

`HOME=/tmp/sft_container_home` keeps git/uv state inside the disposable
container. The actual repo checkout is mounted from the H100 host workspace.
Do not put private Hugging Face tokens into `docker run -e`; Docker users with
inspect privileges may be able to read container environment variables.

## 3.1. tmux Logs And Completion Markers

The demo scripts start inside tmux by default. When a stage finishes, the tmux
pane stays open so the final status remains visible. Press Enter in that pane
when you are done reviewing it.

If a script is launched from a non-interactive shell, it starts the tmux session
detached instead. Attach with the session name printed by the script.

Each tmux-run stage writes stdout/stderr and a completion marker under:

```bash
/workspace/alpamayo-recipes/logs/sft_runs
```

Log and marker names use the run ID and tmux session name:

```text
<run_id>_sft_demo_setup.log
<run_id>_sft_demo_setup.done
<run_id>_sft_demo_setup.failed

<run_id>_sft_demo_stage1_nav.log
<run_id>_sft_demo_stage1_nav.done
<run_id>_sft_demo_stage1_nav.failed
```

Check the newest run after tmux exits or while it is still running:

```bash
cd /workspace/alpamayo-recipes
ls -lt logs/sft_runs | head
tail -f logs/sft_runs/<run_id>_sft_demo_stage1_nav.log
```

For non-interactive automation only, disable the hold prompt:

```bash
SFT_TMUX_HOLD=no scripts/sft_demo_01_stage1_nav_smoke.sh
```

## 4. Prepare Data Only

This step prepares only the nav annotation JSON and required PAI 19-chunk sample
payload under `/data/datasets/physical_ai_av`. It is safe to repeat: existing
files are checked first, and missing chunks are downloaded only when needed.
If Hugging Face authentication is unavailable, the script prompts for `HF_TOKEN`
interactively.

```bash
cd /workspace/alpamayo-recipes
scripts/sft_demo_00_prepare_data.sh
```

This command starts a tmux session by default. Attach with:

```bash
tmux attach -t sft_demo_setup
```

After completion, check:

```bash
ls -lt logs/sft_runs/*_sft_demo_setup.*
```

Expected data-only responsibilities:

- create/sync recipe venv for downloader/status dependencies;
- download and validate `nav_demo_samples.json`;
- create `/data/datasets/physical_ai_av` if missing;
- check existing PAI nav chunks and skip if complete;
- prompt for `HF_TOKEN` only when cached auth is unavailable;
- download missing required chunks.

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

The default uses host GPU `4`, while the container has GPUs `0,1,2,3,4`
available:

```bash
cd /workspace/alpamayo-recipes
scripts/sft_demo_01_stage1_nav_smoke.sh
```

This command starts `tmux` session `sft_demo_stage1_nav` by default. Attach with:

```bash
tmux attach -t sft_demo_stage1_nav
```

After completion, check the marker, log, and checkpoint:

```bash
ls -lt logs/sft_runs/*_sft_demo_stage1_nav.*
tail -80 logs/sft_runs/<run_id>_sft_demo_stage1_nav.log
find /data/alpamayo_sft_artifacts -maxdepth 2 -type d -name 'checkpoint-*' | sort
```

Default execution values are defined inside `scripts/sft_demo_config.sh`:

- `PAI_DIR=/data/datasets/physical_ai_av`
- `ARTIFACT_ROOT=/data/alpamayo_sft_artifacts`
- `SFT_DEFAULT_GPU_IDS=4`
- timestamped Stage 1 output under `/data/alpamayo_sft_artifacts`

Tune these defaults in the script before running if the container-visible GPU
set or data paths differ.

### Stage 1 Single-GPU Capacity Note

The non-root container path was validated through setup and into Stage 1 on
GPU `4` with:

```bash
SFT_RUN_ID=nonroot01_20260622_230835 \
SFT_TMUX_HOLD=no \
STAGE1_MAX_STEPS=1 \
STAGE1_SAVE_STEPS=1 \
scripts/sft_demo_01_stage1_nav_smoke.sh
```

Observed result:

- setup, data, A1 checkpoint verification, tmux logging, and marker creation
  worked as expected;
- Stage 1 loaded the A1 checkpoint and PAI nav demo payload;
- execution used only container-visible GPU `4`;
- the run failed during backward with CUDA OOM on a single H100:
  `Tried to allocate 16.39 GiB` while the process already used about `83.91 GiB`;
- the script wrote `.failed` under `logs/sft_runs`.

This means a single H100 is not enough for the current full-model Stage 1
training settings. Use multiple GPUs with `--gpus`, reduce the training memory
profile, or change the recipe before treating Stage 1 as runnable on one GPU.

### Stage 1 Three-GPU Smoke Result

The same one-step Stage 1 smoke completed on GPUs `2,3,4`:

```bash
SFT_RUN_ID=gpu23401_20260623_082957 \
SFT_TMUX_HOLD=no \
STAGE1_MAX_STEPS=1 \
STAGE1_SAVE_STEPS=1 \
scripts/sft_demo_01_stage1_nav_smoke.sh --gpus 2,3,4
```

Observed result:

- `CUDA_VISIBLE_DEVICES=2,3,4`;
- `nproc_per_node=3`;
- training reached `1/1` step without CUDA OOM;
- `train_loss=1.986689805984497`;
- `checkpoint-1` created;
- `.done` marker written under `logs/sft_runs`.

Output path:
`/data/alpamayo_sft_artifacts/output_stage1_nav_smoke_gpu23401_20260623_082957`.

The checkpoint output was about `132G`: model shards were about `17G`, while the
DeepSpeed `global_step1` state was about `115G`. Keep Stage outputs under
`/data`, not under the repo workspace.

### Stage 1 Twenty-Step TensorBoard Run

The 20-step Stage 1 smoke completed on GPUs `2,3,4` with step-level stdout
logging and TensorBoard enabled:

```bash
SFT_RUN_ID=stage1tb20_20260623_093829 \
SFT_TMUX_HOLD=no \
scripts/sft_demo_01_stage1_nav_smoke.sh --gpus 2,3,4
```

Effective settings:

- `CUDA_VISIBLE_DEVICES=2,3,4`;
- `nproc_per_node=3`;
- `trainer.logging_steps=1`;
- `trainer.report_to=tensorboard`;
- `trainer.save_steps=20`;
- `trainer.max_steps=20`.

Output:

- log: `logs/sft_runs/stage1tb20_20260623_093829_sft_demo_stage1_nav.log`;
- checkpoint: `/data/alpamayo_sft_artifacts/output_stage1_nav_smoke_stage1tb20_20260623_093829/checkpoint-20`;
- TensorBoard event:
  `/data/alpamayo_sft_artifacts/output_stage1_nav_smoke_stage1tb20_20260623_093829/tensorboard/events.out.tfevents.*`;
- final `train_loss=2.1151309609413147`;
- `train_runtime=517.2538` seconds.

Loss did not show a clear downward trend in 20 steps. It stayed around
`2.01-2.18`. This is still useful wiring evidence, but it is not overfit
evidence. The learning rate was still in warmup and extremely small
(`2e-09` to `4e-08`), so this run should not be read as proof that the setup
cannot learn.

TensorBoard was started on the H100 host at port `16006`:

```bash
ssh -L 16006:127.0.0.1:16006 aidev1@192.168.1.44
```

Then open:

```text
http://localhost:16006/
```

### Stage 1 300-Step Overfit Result

The Stage 1 overfit run completed on GPUs `2,3,4` after reducing the demo
warmup from the recipe default:

```bash
SFT_RUN_ID=stage1overfit300_20260623_104948 \
scripts/sft_demo_01_stage1_nav_smoke.sh --gpus 2,3,4
```

Effective settings:

- `trainer.max_steps=300`;
- `trainer.warmup_steps=5`;
- `trainer.logging_steps=1`;
- `trainer.save_steps=300`;
- `trainer.report_to=tensorboard`;
- `CUDA_VISIBLE_DEVICES=2,3,4`.

Observed result:

- log: `logs/sft_runs/stage1overfit300_20260623_104948_sft_demo_stage1_nav.log`;
- checkpoint:
  `/data/alpamayo_sft_artifacts/output_stage1_nav_smoke_stage1overfit300_20260623_104948/checkpoint-300`;
- `.done` marker written under `logs/sft_runs`;
- final step loss: `0.0048`;
- aggregate `train_loss=0.41450476200940706`;
- `train_runtime=5844.3078` seconds.

The loss curve moved from about `2.1` at the start to about `0.005` near the
end. This matches the author's stated intent for the 20-row nav demo: it is an
overfit smoke test proving that the Stage 1 VLM SFT path can learn the tiny
sample set. It is not evidence of generalization.

### Stage 1 Baseline-vs-SFT Inference Check

After the 300-step checkpoint was created, the same nav demo validation path was
run with `evaluate_hf.py` on one H100 GPU. Both runs used the same data,
generation settings, and `max_eval_steps=5`.

Comparison command shape:

```bash
cd /workspace/alpamayo-recipes/recipes/alpamayo1_5_sft
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=4 torchrun --nproc_per_node 1 \
  -m alpamayo1_5_sft.evaluate_hf \
  --config-path pkg://alpamayo1_5_sft/configs \
  --config-name sft_stage1_nav \
  model.checkpoint_path=/data/alpamayo_sft_artifacts/Alpamayo-1.5-10B-A1-format \
  data.val_dataset.local_dir=/data/datasets/physical_ai_av \
  data.val_dataset.annotations_path=/data/alpamayo_sft_artifacts/nav_demo_samples.json \
  trainer.deepspeed=null \
  trainer.report_to=none \
  trainer.per_device_eval_batch_size=1 \
  trainer.dataloader_num_workers=0 \
  evaluate.eval_ckpt=<baseline-or-stage1-checkpoint> \
  evaluate.max_eval_steps=5 \
  +evaluate.metric_runner.metrics.0.num_traj_samples=1 \
  evaluate.metric_runner.metrics.0.num_traj_sets=1 \
  evaluate.metric_runner.metrics.0.temperature=0.1
```

Comparison log:
`logs/sft_runs/stage1infercmp_20260623_043228.log`.

| Metric | Baseline A1 | Stage 1 SFT |
| --- | ---: | ---: |
| `val/metric/ade` | `1.4941` | `0.0968` |
| `val/metric/ade/by_t=3.0` | `0.3032` | `0.0284` |
| `val/metric/corner_distance` | `1.5469` | `0.2529` |
| `val/metric/min_ade/by_t=5.0` | `0.8280` | `0.0634` |

The SFT checkpoint substantially improved trajectory inference on the same
overfit sample distribution: ADE dropped by about 93.5%, 3-second ADE by about
90.6%, and corner distance by about 83.6%. This is the expected effect for the
demo sample: the checkpoint has learned the tiny nav set and is suitable as a
Stage 2 smoke-test input. The result still does not claim held-out validation
quality or broader PAI/a2z readiness.

## Stage 2 Meaning

Stage 2 is the trajectory/action expert stage, not another front-end VLM SFT
stage. `sft_stage2_nav.yaml` uses `models/ar1_5_expert`, which maps to
`TrainableAlpamayoR1`.

The Stage 2 model:

- loads the converted A1-format base checkpoint as `pretrained_model_name_or_path`;
- loads the Stage 1 VLM checkpoint through `stage1_vlm_checkpoint_path`;
- freezes the VLM by default (`cotrain_vlm=false`);
- converts `ego_history_*` and `ego_future_*` into action-space training data;
- trains the action/expert diffusion path using `diffusion.compute_loss_from_pred`.

So Stage 2 is the back-end action model / trajectory diffusion expert training
step. It should run only after the Stage 1 checkpoint has been reviewed as a
usable input.

## Security Notes

- Host persistence is intentionally split between `/home/aidev1/workspace` for
  source and `/data` for large data/artifacts.
- The repo workspace should keep source, logs, and marker files, not large
  checkpoints.
- Git clone/pull happens in the H100 host workspace, not inside the container.
- Do not mount host `$HOME`, `.ssh`, `.gitconfig`, or HF cache into the
  container.
- Do not pass private tokens with `docker run -e`.
- Do not put tokens in `REPO_URL`.
- If credentials are needed, enter them interactively inside the running
  container and delete the container after use.

## Non-Claims

This runbook supports a bounded Stage 1 overfit smoke only. It does not establish
a2z readiness, broad PAI nav readiness, generalization, or autonomous-driving
quality/safety improvement.
