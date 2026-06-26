---
doc_type: note
status: active
created_at: 2026-06-22 15:38:20
updated_at: 2026-06-25 06:58:00
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

## Stage 1 and Stage 2 Artifact Interpretation

The Stage 1 and Stage 2 outputs have different load contracts. Do not treat
every checkpoint directory as the same kind of full Alpamayo 1.5 runtime model.

| Artifact | Current interpretation | Current 20-row mean ADE |
| --- | --- | ---: |
| matched official baseline | 2026-06-18 Alpamayo 1.5 baseline tensor set used for fair overlay comparison | `1.751832239329815` |
| Stage 1-only export | VLM-side nav overfit result exported through the recipes contract; not a standalone official `Alpamayo1_5.from_pretrained` checkpoint | `0.09993635825812816` |
| current Stage 2-only-ish export | Diagnostic result from a Stage 2 action expert run whose VLM stayed baseline-like; do not interpret as `Stage1-improved VLM + Stage2 action expert` | `1.4155602216720582` |

Comparison summary from the matched overlay:

- Stage 1 better than Stage 2: `20/20`;
- Stage 2 better than matched official baseline: `7/20`;
- Stage 1-only strongly overfit the 20-row nav demo;
- current Stage 2-only-ish improved the mean over baseline but was unstable and
  worse than Stage 1 on every row.

The intended Stage 2 experiment was `Stage1-improved VLM + Stage2 action
expert`. The current Stage 2 checkpoint should not be read that way. Its config
included `model.stage1_vlm_checkpoint_path`, but the load log showed:

```text
Loaded 750 VLM tensors from .../checkpoint-300 (missing=750, unexpected=750)
```

A read-only tensor probe then showed that a Stage 2 VLM tensor was bit-exact
with the baseline A1 tensor and not equal to the Stage 1 tensor:

```text
stage2 vs base_a1: max_abs=0.0, mean_abs=0.0, allclose=True
stage2 vs stage1: max_abs=0.0003662109375, mean_abs=8.058943785727024e-05, allclose=False
```

So the current Stage 2 result is useful diagnostic evidence, but it is not the
canonical `Stage1 + Stage2 both SFT` result.

## Stage 2 Meaning

Stage 2 is the trajectory/action expert stage, not another front-end VLM SFT
stage. `sft_stage2_nav.yaml` uses `models/ar1_5_expert`, which maps to
`TrainableAlpamayoR1`.

The intended Stage 2 model contract:

- loads the converted A1-format base checkpoint as `pretrained_model_name_or_path`;
- loads the Stage 1 VLM checkpoint through `stage1_vlm_checkpoint_path`;
- freezes the VLM by default (`cotrain_vlm=false`);
- converts `ego_history_*` and `ego_future_*` into action-space training data;
- trains the action/expert diffusion path using `diffusion.compute_loss_from_pred`.

So Stage 2 is the back-end action model / trajectory diffusion expert training
step. It should run only after the Stage 1 checkpoint has been reviewed as a
usable input.

### Author Demo Reproduction Status

The current checkpoint review should be read as an environment/demo
reproduction check, not as a search for a better training strategy.

Stage 1 status: reproduced.

- README expectation: the 20 nav samples are an overfit smoke set and the loss
  should drop near zero after hundreds of steps.
- Verified artifact:
  `/data/alpamayo_sft_artifacts/output_stage1_nav_smoke_stage1overfit300_20260623_104948/checkpoint-300`.
- Trainer state: `global_step=300`, `epoch=150.0`.
- Loss tail: step 291-300 stayed around `0.0045` to `0.0054`, with step 300
  loss `0.0048`.
- Interpretation: the Stage 1 nav demo path, dataset wiring, nav annotations,
  converted A1-format checkpoint, and recipe training loop are sufficiently
  reproduced for the README-style 20-sample overfit smoke.

Stage 2 status: reproduced through the README/evaluate contract for the default
row13/chunk-2868 eval scope.

- README expectation: train Stage 2 from the converted A1-format base plus a
  Stage 1 checkpoint, then evaluate the Stage 2 checkpoint with
  `alpamayo1_5_sft.evaluate_hf`.
- Important scope detail: `configs/sft_stage2_nav.yaml` uses
  `data.val_dataset.chunk_ids: [2868]`. In the 20-row nav demo annotation file,
  chunk `2868` corresponds only to row13
  (`nav_text="Turn left in 13m"`). It does not evaluate all 20 rows.
- Therefore 20-row compact exports and overlays are the default local demo
  evaluation artifacts, while the README Stage 2 eval contract remains a
  narrower author-style sanity check.
- Attempted official eval command, without new training:

```bash
cd /workspace/alpamayo-recipes
printf 'RUN_EVAL\n' | \
  env SFT_DISABLE_TMUX=yes \
      STAGE2_OUTPUT_DIR=/data/alpamayo_sft_artifacts/output_stage2_nav_overfit300_stage2overfit300_plan_20260623_144245 \
      EVAL_MAX_STEPS=-1 \
      scripts/sft_demo_03_eval_stage2_nav.sh --gpus 4
```

- Log:
  `/workspace/alpamayo-recipes/logs/sft_runs/author_eval_stage2_20260625_063958.log`.
- Result: completed after authenticating the container for the gated
  `nvidia/Cosmos-Reason2-8B` processor/config dependency.
- Dataset filter evidence:
  `Filtered out 19/20 annotated samples whose clip chunks aren't in chunk_ids=[2868]; keeping 1.`
- Official default Stage 2 eval metrics:

| Metric | Value |
| --- | ---: |
| `val/count` | `1.0000` |
| `val/metric/ade` | `1.3345` |
| `val/metric/ade/by_t=3.0` | `0.3792` |
| `val/metric/corner_distance` | `0.6229` |
| `val/metric/min_ade` | `0.6184` |
| `val/metric/min_ade/by_t=0.5` | `0.0056` |
| `val/metric/min_ade/by_t=1.0` | `0.0199` |
| `val/metric/min_ade/by_t=3.0` | `0.1613` |
| `val/metric/min_ade/by_t=5.0` | `0.4211` |

Interpretation:

- The author-style Stage 2 eval path now runs end-to-end in this container.
- Its default scope is row13/chunk 2868, not the full 20-row nav demo set.
- The existing 20-row compact exporter/overlay results remain a broader local
  diagnostic benchmark, not a contradiction of the official eval scope.
- Further Stage 2 training-strategy changes should not be made until this
  distinction is kept explicit in reports and comparisons.

### Loader Contract Pitfall

Before any future Stage 2 run, prove that the Stage 1 VLM weights are actually
loaded into the Stage 2 model. The current failure mode is subtle: Stage 1
checkpoint keys are stored with a `vlm.` prefix, while Stage 2 calls
`load_alpamayo1_vlm(stage1_vlm_checkpoint_path, self.vlm)` on the nested VLM
module. If the loader passes `vlm.*` keys directly into the nested module, the
load can report the same number of missing and unexpected keys and silently leave
the Stage 2 VLM baseline-like.

Required pre-training gate for the next Stage 2 run:

1. Instantiate the Stage 2 model with `stage1_vlm_checkpoint_path`.
2. Confirm the VLM load log does not show broad `missing=750, unexpected=750`
   style mismatch.
3. Probe at least one small VLM tensor and confirm Stage 2 equals Stage 1, not
   the baseline A1 wrapper.
4. Record the exact key, max/mean absolute differences, and `allclose` results
   before starting training.

Example read-only tensor gate:

```bash
python - <<'PY'
import json
from pathlib import Path

import torch
from safetensors import safe_open

key = "vlm.model.language_model.layers.0.input_layernorm.weight"
roots = {
    "base_a1": Path("/data/alpamayo_sft_artifacts/Alpamayo-1.5-10B-A1-format"),
    "stage1": Path("/data/alpamayo_sft_artifacts/output_stage1_nav_smoke_stage1overfit300_20260623_104948/checkpoint-300"),
    "stage2": Path("/data/alpamayo_sft_artifacts/output_stage2_nav_<new_run>/checkpoint-300"),
}

tensors = {}
for name, root in roots.items():
    weight_map = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(str(root / weight_map[key]), framework="pt", device="cpu") as f:
        tensors[name] = f.get_tensor(key).float()

for a, b in [("stage2", "base_a1"), ("stage2", "stage1")]:
    delta = (tensors[a] - tensors[b]).abs()
    print(a, "vs", b, "max_abs", float(delta.max()), "mean_abs", float(delta.mean()), "allclose", bool(torch.allclose(tensors[a], tensors[b])))
PY
```

The expected result for a valid `Stage1 + Stage2` run is that `stage2 vs stage1`
is equal or near-equal for the loaded VLM weights, while `stage2 vs base_a1`
shows the Stage 1 deltas.

### Approved Follow-up Experiment Plan

Do not start runtime training from this section until the PM explicitly sends a
separate start instruction. This section records the approved direction and the
gates that must be satisfied before launch.

#### Phase 1: Stage1-SFT-frozen Stage2 SFT

Question: if the Stage 1-improved VLM is actually inherited by Stage 2, does the
Stage 2 action expert also improve?

Training scope:

- start from the converted A1-format base as the full Stage 2 model base;
- load the Stage 1 SFT VLM weights correctly into the Stage 2 VLM;
- freeze the VLM;
- train only the Stage 2 action/diffusion expert path.

Suggested run/artifact naming:

- run id prefix: `stage1_sft_frozen_stage2_sft_*`;
- output root:
  `/data/alpamayo_sft_artifacts/output_stage1_sft_frozen_stage2_sft_<run_id>`;
- compact export root:
  `/data/alpamayo_sft_artifacts/stage1_sft_frozen_stage2_sft_export_<run_id>`;
- workstation handoff root:
  `/home/user/Workspace/alpamayo1.5/experiments/nav_demo_inference_comparison/<date>/stage1_sft_frozen_stage2_sft_vs_baseline/stage2_export/`.
- reproduction entrypoint:
  `scripts/sft_experiments/run_stage1_sft_frozen_stage2_sft.sh`.

Required pre-training gates:

1. Fix or otherwise prove a safe Stage 1 VLM load path for nested Stage 2 VLM
   loading.
2. Report the Stage 1 VLM load log. Broad `missing=750, unexpected=750` is a
   launch blocker unless explicitly explained by key mapping evidence.
3. Prove that a Stage 2 model VLM tensor equals or is allclose to the Stage 1
   checkpoint tensor.
4. Prove that the same Stage 2 model VLM tensor is not still bit-exact baseline
   A1.
5. Report frozen parameter scope and trainable parameter scope, including
   parameter counts.

Success criteria:

- full 20-row nav demo export uses the common sampling settings below;
- no single row is used as a pass/fail gate for the demo result;
- mean ADE and row-wise counts improve over the current Stage2-only-ish result;
- if the action expert can consume the improved VLM context, results should move
  closer to Stage 1-only than to the current Stage2-only-ish artifact.

Failure interpretation:

- failing the loader/equality gate means the runtime is not testing Stage1 plus
  Stage2 and must not train;
- passing the loader gate but failing rollout metrics points to Stage 2
  objective/eval contract difficulty, exposure bias, insufficient action expert
  capacity, or sampling behavior rather than the previously observed loader bug.

#### Phase 2: Stage1 plus Stage2 Joint SFT

Question: if selected VLM layers and the Stage 2 action expert are updated
together, does 20-row overfit and rollout stability improve further?

This phase requires a separate PM approval after Phase 1 review. The starting
point is still a decision:

- baseline start: cleanest reset, but mixes Stage 1 and Stage 2 learning causes;
- Stage 1 checkpoint start: preserves the known nav overfit effect and lets the
  action path adapt around it;
- Phase 1 continuation: fastest, but inherits any Phase 1 bias or mistakes.

Current preferred direction after Phase 1 success: start from the Stage 1
checkpoint and update selected VLM layers plus the action expert. Full VLM
unfreeze is higher risk and needs a separate reason.

Required gates:

1. PM approval after Phase 1 metrics and visuals are reviewed.
2. Explicit trainable parameter scope, including which VLM layers are unfrozen.
3. Stage 1 nav retention gate to catch catastrophic forgetting.
4. Same full 20-row export and matched overlay comparison as Phase 1.

Suggested run/artifact naming:

- run id prefix: `stage1_stage2_joint_sft_*`;
- output root:
  `/data/alpamayo_sft_artifacts/output_stage1_stage2_joint_sft_<run_id>`;
- compact export root:
  `/data/alpamayo_sft_artifacts/stage1_stage2_joint_sft_export_<run_id>`;
- workstation handoff root:
  `/home/user/Workspace/alpamayo1.5/experiments/nav_demo_inference_comparison/<date>/stage1_stage2_joint_sft_vs_baseline/stage2_export/`.

#### Common Evaluation and Export Contract

All Phase 1 and Phase 2 comparisons use:

- row set: `/data/alpamayo_sft_artifacts/nav_demo_samples.json`, all 20 rows;
- `nav_text`: annotation text unchanged;
- seed: `42`;
- `top_p=0.98`;
- `temperature=0.6`;
- `diffusion_temperature=0.6` when applicable;
- `num_traj_samples=1`;
- `num_traj_sets=1` when applicable;
- `max_generation_length=256`;
- dtype: `bfloat16`;
- metrics: mean and median ADE/minADE, endpoint distance, corner distance when
  available, row-wise better counts, best/worst rows, and obvious failure
  clusters.

Historical row07 reference values remain useful as one reproducibility note among
the 20 rows. They are not a gate, special report, or primary decision row:

| Source | Endpoint xy | ADE |
| --- | --- | ---: |
| official baseline | `[33.33718490600586, -3.5824496746063232]` | `0.5370676517486572` |
| Stage 1-only export | `[32.469303131103516, -2.4979147911071777]` | `0.011621384881436825` |
| current Stage2-only-ish | `[33.731842041015625, -2.083317279815674]` | `0.6909735202789307` |

Every compact export handed to the visualization thread must keep the existing
schema:

- `manifest.json`;
- `annotations_snapshot.json`;
- `<model_name>/results.jsonl`;
- `<model_name>/predictions.npz`;
- `<model_name>/summary.json`.

The NPZ keys should remain compatible with the current renderer:

- `row_XX/pred_xyz`;
- `row_XX/pred_rot`;
- `row_XX/ego_future_xyz`;
- `row_XX/ego_future_rot`;
- `row_XX/ego_history_xyz`;
- `row_XX/ego_history_rot`.

Before copying model weights or any large artifact to another machine, report
the size, available target space, and target path for PM approval. Compact
prediction exports are the preferred visualization handoff.

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

This runbook supports bounded nav-demo SFT smoke tests and diagnostic
comparisons only. It does not establish a2z readiness, broad PAI nav readiness,
generalization, or autonomous-driving quality/safety improvement.
