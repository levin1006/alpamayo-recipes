---
doc_type: plan
status: active
plan_id: track-c-pai-stage1-overfit-smoke
version: 1
canonical: true
created_at: 2026-06-22 12:56:51 KST
approved_at: 2026-06-22 12:56:51 KST
root_plan: docs/2026-06-17 [Plan] Track C a2z SFT Feasibility and Implementation Readiness.md
supersedes: docs/2026-06-22 [Plan] Track C PAI SFT Demo Gate Before a2z.md
superseded_by:
root_task: docs/2026-06-22 [Task] Track C PAI 20 Sample Stage 1 Overfit Smoke Root Task.md
revision_type: stage1_overfit_smoke_after_p0_p1
revision_reason: Verify the expected 20-sample overfit behavior of the official PAI nav SFT recipe before returning to a2z planning
source_repo: /home/user/Workspace/alpamayo-recipes
related_inference_repo: /home/user/Workspace/alpamayo1.5
dataset_root: /mnt/zfs_pool/physical_ai_av
annotations_path: /home/user/Workspace/alpamayo1.5/notebooks/nav_demo_samples.json
---

# Track C PAI 20 Sample Stage 1 Overfit Smoke

## Document Frame

Purpose: define the next Track C gate as a bounded Stage 1 overfit smoke on the
official Alpamayo 1.5 PAI nav demo samples.

Primary reader: the user, the Track C PM/manager session, and the separate
execution thread that will run the experiment.

Decision question: can the official recipe train on the 20 annotated PAI nav
demo rows well enough to show the expected overfit signal under a strict cap?

Exclusion scope: this Plan does not claim generalization, broad PAI readiness,
a2z readiness, autonomous-driving quality improvement, unbounded training,
automatic Stage 2, source dataset mutation, or git push.

## Current Decision

The user correctly identified that training and testing on the same 20 samples
should not be treated as a real model-quality evaluation. The value of this
step is narrower:

```text
If the recipe cannot overfit the 20 rows, the training path is probably wired
incorrectly. If it can, the result proves only that the PAI nav Stage 1 SFT
path is executable and learnable on the tiny demo payload.
```

Therefore the expected result is not "the model is better". The expected result
is:

- finite loss at launch;
- loss decreasing under a bounded run;
- a checkpoint artifact written by the trainer;
- optional same-20-row inference/ADE comparison classified as an overfit
  confirmation only;
- no claim that this transfers to a2z or held-out PAI data.

## Inputs

Use the already established demo payload:

- PAI dataset root: `/mnt/zfs_pool/physical_ai_av`
- nav annotations: `/home/user/Workspace/alpamayo1.5/notebooks/nav_demo_samples.json`
- sample shape: 20 rows, 19 unique clips
- base checkpoint input: raw `nvidia/Alpamayo-1.5-10B`
- SFT checkpoint path required by Stage 1: converted A1-format checkpoint

## Script Baseline

The script surface is consolidated around the README step sequence:

```bash
scripts/a15_sft_readme_00_setup_env.sh
scripts/a15_sft_readme_01_download_pai_nav_chunks.sh
scripts/a15_sft_readme_02_download_checkpoint.sh
scripts/a15_sft_readme_03_convert_checkpoint_to_a1.sh
scripts/a15_sft_readme_04_verify_a1_checkpoint.sh
scripts/a15_sft_readme_05_stage1_nav_smoke.sh
scripts/a15_sft_readme_06_stage2_nav_smoke.sh
scripts/a15_sft_readme_07_eval_stage2_nav.sh
```

All step scripts default to tmux. The Stage 1 script requires explicit
confirmation text before training starts.

Retired duplicate wrappers:

- `scripts/setup_pai_sft_demo_env.sh`
- `scripts/run_pai_sft_demo_p0_p1.sh`
- `scripts/check_pai_nav_status.sh`
- `scripts/plan_pai_nav_download.sh`
- `scripts/start_pai_nav_download_tmux.sh`

The core utility scripts remain:

- `scripts/download_pai.py`
- `scripts/convert_checkpoint.py`
- `scripts/check_pai_download_status.py`
- `scripts/plan_pai_nav_download.py`

## Execution Gate

The execution thread may run only the following sequence unless the user gives
new approval in that thread:

1. Verify environment setup.
2. Verify or download PAI nav chunks.
3. Verify or download the raw Alpamayo 1.5 checkpoint.
4. Convert the raw checkpoint to A1-format.
5. Verify A1-format checkpoint metadata.
6. Run bounded Stage 1 nav SFT smoke.
7. Report results.

Stage 2 and evaluation are intentionally not part of the first required gate.
They remain optional follow-up work after Stage 1 review.

## Stage 1 Constraints

Required defaults:

- `WANDB_MODE=disabled`
- `trainer.report_to=none`
- `NPROC_PER_NODE=1` unless the user changes it
- `CUDA_VISIBLE_DEVICES=0` unless the user changes it
- strict `trainer.max_steps`, default `20`
- fresh output directory under `/mnt/zfs_pool/alpamayo_sft_artifacts`
- `trainer.save_total_limit=1`
- absolute DeepSpeed config path

Stop on:

- missing or unverified A1-format checkpoint;
- Hydra instantiation failure;
- CUDA OOM;
- NaN or inf loss;
- accidental external logging;
- output path collision that could overwrite a useful run;
- request to continue into Stage 2 without separate review.

## Review Questions

The execution report must answer:

1. Did the A1-format checkpoint verify as `model_type: alpamayo_r1` with
   `architectures: ["AlpamayoR1"]`?
2. Did Stage 1 instantiate the model, dataset, collator, and trainer?
3. Did loss remain finite?
4. Did loss decrease enough to support the narrow "overfit smoke" claim?
5. Was a checkpoint artifact written?
6. If same-20-row inference/ADE is run later, is it explicitly labeled as
   overfit confirmation rather than model-quality evidence?

## Acceptance Criteria

This Plan is successful if the execution thread returns a Korean report with:

- exact commands or scripts run;
- exact environment overrides;
- A1-format checkpoint path;
- output directory;
- loss samples or log excerpt;
- checkpoint artifact path if created;
- pass/fail classification;
- Evidence, Inference, Unknowns separated;
- explicit non-claims about generalization, a2z readiness, and broad PAI
  readiness.

## Reporting

The execution thread should report completion, failure, blocker, or user
decision needs to this Track C manager session:

`019ecfca-468c-7682-b75e-aadca21dfe86`

If PM routing is also needed, use the current PM reporting target:

`019eed40-e996-75b3-aa65-916422226066`
