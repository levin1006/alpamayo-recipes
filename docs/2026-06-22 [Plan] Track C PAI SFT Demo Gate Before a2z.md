---
doc_type: plan
status: superseded
plan_id: track-c-pai-sft-demo-gate-before-a2z
version: 1
canonical: false
created_at: 2026-06-22 11:17:55 KST
approved_at: 2026-06-22 11:18:19 KST
root_plan: docs/2026-06-17 [Plan] Track C a2z SFT Feasibility and Implementation Readiness.md
supersedes:
superseded_by: docs/2026-06-22 [Plan] Track C PAI 20 Sample Stage 1 Overfit Smoke.md
root_task: docs/2026-06-22 [Task] Track C PAI SFT Demo Root Task.md
revision_type: demo_gate_before_a2z_runtime
revision_reason: Verify the official PAI nav demo SFT path before returning to a2z readiness work
source_repo: /home/user/Workspace/alpamayo-recipes
related_inference_repo: /home/user/Workspace/alpamayo1.5
dataset_root_candidate: /mnt/zfs_pool/physical_ai_av
annotations_path: /home/user/Workspace/alpamayo1.5/notebooks/nav_demo_samples.json
---

# Track C PAI SFT Demo Gate Before a2z

## Document Frame

Purpose: define a narrow, reviewable gate for checking whether the official PAI
navigation demo can exercise the Alpamayo 1.5 SFT recipe path before Track C
returns to a2z SFT readiness.

Primary reader: the user and the separate Track C 총괄 session that will review
the result and decide whether a2z work may resume.

Decision question: can this execution session run a bounded official PAI
nav-demo preflight and no-training batch proof, then request separate approval
for a minimal Stage 1 overfit smoke?

Exclusion scope: this Plan does not authorize a2z training, broader PAI nav
training claims, synthetic nav ground truth generation, future-trajectory based
route reconstruction, unbounded training, automatic Stage 2, git commit, or git
push.

## Approval Boundary

This Plan is approved and active for the P0/P1 gate.

Approval authorizes only:

- P0 preflight checks
- P1 no-training dataset, processor, and collator batch proof
- creation or update of the matching Task execution log

Plan approval does not authorize Stage 1 training. Stage 1 requires a second
explicit user approval after P1 passes. Stage 2 and evaluation require another
explicit approval after the Stage 1 report.

## Current Evidence

Evidence:

- `docs/2026-06-17 [Plan] Track C a2z SFT Feasibility and Implementation Readiness.md`
  is active and canonical for a2z readiness, but explicitly excludes training,
  checkpoint creation, and model loading.
- `docs/2026-06-17 [Report] PAI Dataset Nav Annotation Status.md` identifies
  `/mnt/zfs_pool/physical_ai_av` as the stronger PAI root for the bundled nav
  smoke payload, with required 19 nav chunks downloaded for the checked
  components at report time.
- The same report records
  `/home/user/Workspace/alpamayo1.5/notebooks/nav_demo_samples.json` as 20 rows
  across 19 unique clips, with required fields `clip_id`, `t0_relative`, and
  `nav_text`.
- `recipes/alpamayo1_5_sft/README.md` describes these 20 PAI nav samples as an
  overfit smoke test for Stage 1 SFT, not as released-model training data.
- `recipes/alpamayo1_5_sft/configs/sft_stage1_nav.yaml` points train and val to
  `PAIDatasetWithNav`, leaves `local_dir` and `annotations_path` as required
  overrides, uses nav processor config, and sets `num_train_epochs: 700`.
- `recipes/alpamayo1_5_sft/configs/sft_base.yaml` defaults `report_to: none`,
  `save_total_limit: 2`, and a relative DeepSpeed path
  `configs/deepspeed/zero2.json`.
- `src/alpamayo/data/pai_nav.py` loads each annotation row as a sample, uses
  `t0_relative` as `t0_us`, and injects `nav_text` into the sample.
- `recipes/alpamayo1_5_sft/configs/vla_processor/nav.yaml` includes `route` in
  the input component order and keeps `label_components: ["traj_future"]`.
- `src/alpamayo/chat_template/components.py` wraps `nav_text` in route tokens
  when present.
- `src/alpamayo/processor/qwen_processor.py` creates `labels_mask` from
  `label_components` during collate, and sets an all-false mask for generation
  mode.

Inference:

- The official PAI demo can be used as a recipe wiring and overfit-smoke gate
  only. It cannot prove broad PAI nav training readiness or a2z readiness.
- The first executable proof should stop at dataset instantiation, one train
  sample or batch, one val sample or batch, route-text presence, and label-mask
  placement.
- The Stage 1 config is unsafe to launch as-is for this gate because it has no
  bounded `max_steps`, no resolved checkpoint path, no resolved data paths, and
  a relative DeepSpeed config path that may break after Hydra changes cwd.

Unknowns:

- Current filesystem state of `/mnt/zfs_pool/physical_ai_av` must be rechecked.
- Current row and unique-clip counts in the nav JSON must be rechecked.
- Current converted Alpamayo 1.5 A1-format checkpoint path is unknown.
- Current GPU count, GPU memory, recipe `uv` environment state, and DeepSpeed
  config path validity are unknown.
- Whether Stage 1 can fit locally without OOM is unknown until a capped run is
  separately approved.

## Gate Sequence

### P0. Preflight

Goal: prove that the local runtime inputs are concrete enough to attempt a
no-training batch proof.

Checks:

- PAI root exists and required metadata is present.
- Required nav 19 chunks have the four camera components,
  `camera_intrinsics`, `sensor_extrinsics`, and `egomotion`.
- Nav JSON has 20 rows and 19 unique clips.
- Configured train/val chunks match the annotated clips.
- A converted Alpamayo 1.5 A1-format checkpoint path is either found or marked
  unresolved. Discovery must not load model weights.
- `uv` recipe environment can import the recipe and shared package.
- GPU state is recorded with `nvidia-smi`.
- DeepSpeed config is resolved as an absolute path.
- W&B remains disabled through `WANDB_MODE=disabled` and
  `trainer.report_to=none`.

Stop rules:

- Stop before P1 if PAI root is missing, required nav payload is incomplete, nav
  JSON shape does not match 20 rows / 19 unique clips, or the recipe env cannot
  import the dataset/processor code.
- Stop before any training if checkpoint path or A1-format conversion state is
  unknown.

### P1. No-Training Batch Proof

Goal: prove that the official PAI nav demo reaches the dataset, processor, and
collator contract without model loading, training, checkpoint creation, or
dataset mutation.

Checks:

- Instantiate `PAIDatasetWithNav` for train and val using
  `/mnt/zfs_pool/physical_ai_av` and the nav JSON.
- Load at least one train sample and one val sample.
- Confirm `nav_text` exists on the loaded sample.
- Confirm tokenized text contains route markers or route content from
  `nav_text`.
- Collate one train batch with `generation_mode: false`.
- Collate one val batch with `generation_mode: true`.
- Confirm train `labels_mask` has true values only for `traj_future` plus the
  assistant end-of-message token path used by the collator.
- Confirm val `labels_mask` is all false.

Stop rules:

- Stop before Stage 1 if sample load fails, route text is absent from the
  actual tokenized input, collate fails, or the label mask cannot be tied to
  `traj_future`.
- Do not treat P1 pass as learning, model quality, safety, PAI nav readiness, or
  a2z readiness.

### P2. Stage 1 Bounded Overfit Smoke

Goal: after separate user approval, test whether the official Stage 1 nav
training path can run under a strict cap and produce finite loss/log/checkpoint
artifacts.

Required launch constraints:

- Use `sft_stage1_nav`.
- Override `data.train_dataset.local_dir` and `data.val_dataset.local_dir` to
  `/mnt/zfs_pool/physical_ai_av`.
- Override both annotation paths to
  `/home/user/Workspace/alpamayo1.5/notebooks/nav_demo_samples.json`.
- Override `model.checkpoint_path` to the verified A1-format checkpoint.
- Set a strict `trainer.max_steps` cap.
- Keep `WANDB_MODE=disabled` and `trainer.report_to=none`.
- Resolve `trainer.deepspeed` to an absolute existing path.
- Use a new output directory that cannot collide with existing outputs.
- Set `trainer.save_total_limit=1`.
- Record the exact command, overrides, output path, loss/log path, and
  checkpoint path.

Suggested first cap:

```bash
trainer.max_steps=20
trainer.logging_steps=1
trainer.save_steps=20
trainer.save_total_limit=1
```

Stop rules:

- Stop on NaN/inf loss, OOM, missing checkpoint path, unbounded config, external
  logging activation, or output directory collision.
- Do not continue into Stage 2 automatically.

### P3. Review Report

Goal: document what the Stage 1 smoke proved and did not prove.

Required report fields:

- Evidence, Inference, Unknowns separated.
- Preflight command and result summary.
- No-training proof command and result summary.
- If Stage 1 is approved and run: exact launch command, overrides, output
  directory, loss samples, checkpoint directory, failure or pass classification.
- Explicit non-claims: no broad PAI nav readiness, no a2z readiness, no
  generalization claim.
- Question to the user: whether to stop, rerun Stage 1 with adjusted cap, or
  request optional Stage 2/eval.

### P4. Optional Stage 2 and Eval

Goal: only after user approval, run a bounded Stage 2/eval path from the Stage 1
checkpoint.

Constraints:

- Use `sft_stage2_nav`.
- Use the verified Stage 1 checkpoint.
- Use a strict step cap and separate output directory.
- Run `evaluate_hf.py` only after Stage 2 artifact existence is verified.
- Generate GT vs prediction visualization only if the evaluation path succeeds
  and required visualization code/data are available.

## Acceptance Criteria

Plan approval acceptance:

- The user accepts that this gate is PAI demo-only and separate from a2z.
- The user accepts that Plan approval only opens P0 and P1.
- The user accepts that Stage 1, Stage 2, and eval each require separate
  approval.

Preflight acceptance:

- PAI root, nav JSON, nav chunk payload, recipe env, GPU state, DeepSpeed path,
  W&B-disabled policy, and checkpoint discovery state are recorded.
- Any missing checkpoint state is treated as a Stage 1 blocker, not ignored.

No-training batch proof acceptance:

- Train and val dataset objects instantiate.
- At least one train and one val item load.
- Route/nav text is present in the actual processed input.
- Train label mask reaches the `traj_future` supervision path.
- Val generation-mode mask is all false.
- No model weights are loaded and no checkpoint/output training artifact is
  created.

Stage 1 acceptance, if separately approved:

- The command is capped by `max_steps` or an equivalent hard stop.
- The output directory is new and recorded.
- External logging is disabled.
- Loss/log/checkpoint artifacts are recorded.
- Result is classified as recipe wiring/overfit smoke only.

## Reporting Language

All execution reports for this Plan must be in Korean and must separate:

- Evidence
- Inference
- Unknown
- Stop condition or next approval question

The phrase "20-row demo overfit success" must not be used as evidence for "PAI
nav training readiness" or "a2z SFT readiness".

## Admin Changelog

- 2026-06-22 12:56:51 KST: administrative supersession only. This P0/P1 gate
  Plan is superseded by
  `docs/2026-06-22 [Plan] Track C PAI 20 Sample Stage 1 Overfit Smoke.md`
  after P0/P1 passed and the user selected the next gate as a bounded
  20-sample overfit smoke.
