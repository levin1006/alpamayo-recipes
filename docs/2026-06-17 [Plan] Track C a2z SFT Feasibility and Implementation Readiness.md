---
doc_type: plan
status: active
plan_id: track-c-a2z-sft-readiness
version: 1
canonical: true
created_at: 2026-06-17 14:46:57 KST
approved_at: 2026-06-17 14:49:50 KST
root_plan: docs/alpamayo1_5_handoff/2026-06-15 [Plan] Alpamayo 1.5 Research and Development v6.md
supersedes:
superseded_by:
root_task: docs/2026-06-17 [Task] Track C a2z SFT Readiness Root Task.md
revision_type: initial_feasibility_to_implementation_plan
revision_reason: Prepare a2z SFT training by first making the SFT structure understandable and reviewable before any training approval
source_repo: /home/user/Workspace/alpamayo-recipes
related_inference_repo: /home/user/Workspace/alpamayo1.5
dataset_root: /mnt/ssd1tb/datasets/00_data_pre_autolabel
---

# Track C a2z SFT Feasibility and Implementation Readiness Plan

## Document Frame

Purpose: define how Track C should move from a2z SFT feasibility into an
implementation-ready training preparation plan, without starting training before
the user understands and reviews the SFT structure, data contract, and training
principles.

Primary reader: the user reviewing whether a2z SFT is understood well enough to
approve implementation and, later, a micro training run.

Decision question: what must be proven, explained, implemented, and reviewed
before a2z data may be used with the Alpamayo 1.5 SFT recipe?

Exclusion scope: this Plan does not approve training, create checkpoints, load
model weights, mutate source a2z data, copy the dataset, run Track B inference,
modify the Alpamayo 1.5 dashboard/runtime repo, commit, or push.

## Current Conclusion

a2z SFT should start as a data-materialization and learning-review project, not
as a training run.

The official recipe already has a usable training spine:

```text
Hydra config
  -> train_hf.py
  -> model, train_dataset, val_dataset, collate_fn
  -> HuggingFace Trainer.train()
  -> model.forward(tokenized_data, ego_history_*, ego_future_*, labels_mask)
```

The missing piece is not the trainer. The missing piece is an a2z dataset path
that emits the same semantic sample fields as the PAI dataset path, passes the
official VLA processor/collator, enforces split/leakage rules, and gives the
user a clear explanation of what is supervised.

Recommended first training target, if implementation is later approved:
**Stage 1/default trajectory SFT**, not navigation-conditioned SFT. The default
trajectory processor supervises `traj_future` without route text. The `nav`
variant adds a `route` component and therefore needs a separate route-label
policy that a2z does not currently provide.

## Evidence Summary

### SFT execution spine

Evidence:

- `recipes/alpamayo1_5_sft/train_hf.py:38-90` instantiates the model, train
  dataset, val dataset, collate function, callbacks, and `ReasoningVLA_Trainer`,
  then calls `trainer.train()`.
- `recipes/alpamayo1_5_sft/trainer.py:27-152` extends HuggingFace
  `TrainingArguments` and `Trainer`, with optional learning-rate multipliers.
- `recipes/alpamayo1_5_sft/configs/sft_base.yaml:1-67` defines the base dataset
  targets, collator target, trainer settings, output path, evaluation metric
  runner, DeepSpeed, and gradient checkpointing.

Inference:

- The least invasive a2z integration point is a new dataset/materializer target
  under `src/alpamayo/data`, plus one or more recipe config files that point
  `data.{train,val}_dataset._target_` at that new dataset.

### Dataset and processor contract

Evidence:

- `src/alpamayo/data/pai.py:31-163` shows that `PAIDataset` loads a sample,
  squeezes `ego_*` tensors, optionally adds reasoning data, and then stores
  `tokenized_data` from the VLA preprocessor.
- `src/alpamayo/data/pai_nav.py:41-122` shows that `PAIDatasetWithNav` is just
  a PAI sample path plus annotation-driven `t0_relative` and `nav_text`.
- `src/alpamayo/processor/qwen_processor.py:172-228` collates samples, tokenizes
  text, and builds `labels_mask`.
- `src/alpamayo/processor/qwen_processor.py:230-301` sorts images by camera
  index, builds the chat template, processes images, expands image tokens, and
  returns `tokenized_data`.
- `src/alpamayo/chat_template/components.py:98-179` shows image, history, and
  future trajectory component construction.
- `src/alpamayo/utils/get_label_mask.py:50-80` masks only selected label
  components between special start/end tokens.

Inference:

- A valid a2z dataset item must include at least `image_frames`,
  `camera_indices`, `relative_timestamps`, `ego_history_xyz`,
  `ego_history_rot`, `ego_future_xyz`, and `ego_future_rot`, and must optionally
  carry source metadata for review/debug. `tokenized_data` should be produced by
  the same official preprocessor path, not by custom string assembly.

### Stage 1, Stage 2, and labels

Evidence:

- `recipes/alpamayo1_5_sft/configs/vla_processor/default.yaml:1-9` defines
  default trajectory SFT as `image`, `traj_history`, `prompt`, `traj_future`,
  with `traj_future` as the label component.
- `recipes/alpamayo1_5_sft/configs/vla_processor/nav.yaml:1-7` adds `route`,
  camera IDs, and frame numbers, while still supervising `traj_future`.
- `recipes/alpamayo1_5_sft/models/sft_base_model.py:43-74` tokenizes future
  trajectory from `ego_history_*` and `ego_future_*`.
- `recipes/alpamayo1_5_sft/models/sft_alpamayo_r1.py:61-75` converts future
  trajectory tensors into action-space diffusion training data.
- `recipes/alpamayo1_5_sft/models/sft_alpamayo_r1.py:97-189` computes the
  forward loss from tokenized data, trajectory tensors, label mask, VLM outputs,
  expert outputs, and diffusion loss.
- `recipes/alpamayo1_5_sft/configs/sft_stage2_nav.yaml:1-38` uses the expert
  model, turns DeepSpeed off, and expects a Stage 1 VLM checkpoint path.

Inference:

- Stage 1/default trajectory is the safest first a2z target because it avoids
  unapproved `nav_text` generation while exercising the official trajectory
  supervision path.
- Stage 2 should remain a later gate because it depends on a Stage 1 checkpoint
  policy and trains the action expert, increasing runtime and checkpoint risk.

### a2z source contract

Evidence:

- `docs/alpamayo1_5_handoff/2026-06-15 [Note] Track B a2z Alpamayo Input Data Reference Note.md`
  defines a2z user-facing identity as `full_scene_id + t0 window + camera set`
  and says `ego_future_*` is ground truth, not inference input.
- The same note defines the front-three camera mapping as `CAM_FRONT_LEFT`,
  `CAM_FRONT`, `CAM_FRONT_RIGHT` to Alpamayo indices `[0, 1, 2]`.
- The same note defines table traversal through `scene.json`, `sample.json`,
  `sample_data.json`, `ego_pose.json`, `calibrated_sensor.json`, and
  `sensor.json`.
- A representative local a2z log has `v1.0-trainval/{scene,sample,sample_data,
  ego_pose,calibrated_sensor,sensor}.json` plus camera images under
  `samples/<channel>/`.
- Representative JSON rows show timestamps stored as strings and filenames with
  Windows-style separators, so parsing must normalize timestamps and paths.

Inference:

- Track B runtime artifacts are valuable evidence and selection metadata, but
  SFT labels must be rebuilt from source a2z records. `pred_*` output must never
  become supervised target data.

## User Understanding Gate

Training cannot start until the user can review and explain the following in
their own words:

1. What Stage 1 trains: the VLM-side supervised path that sees images, history,
   prompt text, and label-masked future trajectory tokens.
2. What Stage 2 trains: the trajectory/action expert path, using a Stage 1 VLM
   checkpoint and diffusion-style future trajectory training data.
3. What is input versus label: `image_frames`, `camera_indices`,
   `relative_timestamps`, and `ego_history_*` are input-side context;
   `ego_future_*` is supervision/evaluation ground truth.
4. Why Track B predictions are forbidden as labels: `pred_xyz` and `pred_rot`
   are pretrained outputs, not human/source truth.
5. Why split policy matters: neighboring windows and same-log scenes can leak
   nearly identical visual and trajectory context across train/val/test.
6. What the first no-training check proves: only dataset/processor/collator
   compatibility, not learning quality.
7. What a micro training smoke would prove later: trainer/runtime wiring and
   finite loss under a strict cap, not model improvement or safety.

Review pass condition:

- The user accepts a written explanation of the SFT dataflow.
- The user answers review questions about input/label boundaries, Stage 1 vs
  Stage 2, and split leakage.
- Any misunderstood item is corrected in this Plan or a successor canonical
  Plan before implementation proceeds.

## Feasibility Plan

### F0. Training path explanation and review artifact

Goal: make the SFT structure understandable before implementation.

Actions:

- Write a user-facing explanation of the SFT path:
  `config -> dataset -> processor -> collator -> trainer -> model.forward`.
- Include a concrete sample-field mapping:
  `a2z source record -> sample dict -> tokenized_data/batch -> model loss`.
- Include review questions and answer key for the user.

Stop rule:

- Do not move to implementation if the user cannot distinguish `ego_future_*`
  as label/GT from model input context.

### F1. a2z source inventory and row identity

Goal: define a stable source manifest for no-training materialization.

Actions:

- Use `source_dataset_dir`, `source_scene_name`, `scene_token`,
  `full_scene_id`, `t0_sample_index`, `t0_timestamp_ns`, and `camera_set` as row
  identity.
- Treat `full_scene_id` as the user-facing scene key.
- Normalize `sample_data.filename` path separators before resolving images.
- Parse timestamp fields as integers.

Stop rule:

- Stop if a representative row cannot resolve all front-three image history,
  16 history poses, and 64 future poses without exceeding the residual policy.

### F2. Split and leakage policy

Goal: make training/validation boundaries explicit before any data is
materialized.

Actions:

- Default to log-level split by `source_dataset_dir`.
- Enforce that a `full_scene_id` never crosses splits.
- Permit scene-level split only for loader/debug smoke, and label it as
  non-generalization evidence.
- Keep user-reviewed Track B showcase scenes out of train if they are intended
  for qualitative before/after review.

Stop rule:

- Stop if any split manifest places the same log in multiple real SFT splits.

### F3. No-training materialization check

Goal: prove a2z rows can pass the official recipe's sample, processor, and
collate contract without loading model weights or creating checkpoints.

Actions:

- Create a tiny manifest, initially 20-40 rows from 2-4 logs.
- Load images from source paths; do not copy raw images.
- Rebuild `ego_history_xyz`, `ego_history_rot`, `ego_future_xyz`, and
  `ego_future_rot` from source `ego_pose`.
- Produce `tokenized_data` through the official VLA preprocessor.
- Run dataset `__getitem__` and collate on small batches.

Pass evidence:

- Sample dict contains all required raw tensors and source metadata.
- `tokenized_data` includes text and image processor outputs.
- Collate returns `tokenized_data.input_ids` and `labels_mask`.
- `labels_mask` covers `traj_future` and not history/image tokens.
- No model load, no trainer, no checkpoint, no dataset mutation.

### F4. Micro SFT smoke proposal

Goal: define the narrow runtime envelope for a later user-approved training
smoke.

Actions:

- Stage 1/default trajectory first.
- Use 20 train rows and 5-10 val rows from different logs where possible.
- Disable external logging unless explicitly approved.
- Set `max_steps` or a very small epoch/step cap.
- Set `save_total_limit: 1` and an approved scratch/output path outside the
  source dataset.
- Treat first local run as fit/OOM discovery if hardware is below the recipe's
  8-GPU validation envelope.

Stop rule:

- Stop before launch if checkpoint path, scratch path, W&B policy, GPU/runtime
  envelope, or user understanding review is unresolved.

## Implementation Plan

Implementation should start only after the feasibility plan and user
understanding gate are accepted.

### I0. Add a2z manifest schema and validator

Files likely touched:

- `src/alpamayo/data/a2z_manifest.py`
- `tests/test_a2z_manifest_static.py`

Responsibilities:

- Load a JSON/JSONL manifest with row identity, split, source paths, timestamps,
  scene metadata, and camera-set policy.
- Validate required fields, split ownership, duplicate row keys, timestamp
  integer conversion, and path normalization.
- Reject rows containing `pred_xyz`, `pred_rot`, or Track B prediction-derived
  label fields.

Verification:

- CPU unit tests for valid manifest, duplicate row rejection, split leakage
  rejection, and forbidden `pred_*` label rejection.

### I1. Add a2z source table loader

Files likely touched:

- `src/alpamayo/data/a2z_tables.py`
- `tests/test_a2z_tables_static.py`

Responsibilities:

- Read `v1.0-trainval` JSON tables from one a2z log directory.
- Index scenes, samples, sample_data, ego_pose, calibrated_sensor, and sensor.
- Resolve camera sample paths under the source log.
- Normalize backslash filenames to platform paths.

Verification:

- CPU tests with tiny synthetic JSON fixtures, not the full dataset.
- Optional read-only smoke on one real a2z log that only counts/validates fields.

### I2. Add a2z trajectory materializer

Files likely touched:

- `src/alpamayo/data/a2z_materializer.py`
- `tests/test_a2z_materializer_static.py`

Responsibilities:

- Select image history targets at `[-300, -200, -100, 0] ms`.
- Select 16 history poses ending at `t0`.
- Select 64 future poses after `t0`.
- Apply the `T_E0_E(tau) = inverse(T_G_E(t0)) @ T_G_E(tau)` transform.
- Emit tensors matching the PAI/Alpamayo sample contract.
- Record residuals and fail/warn status.

Verification:

- Synthetic transform tests where expected local-frame positions are known.
- Residual threshold tests for pass/warn/fail classification.
- Shape tests for front-three output.

### I3. Add `A2ZTrajectoryDataset`

Files likely touched:

- `src/alpamayo/data/a2z.py`
- `recipes/alpamayo1_5_sft/configs/sft_stage1_a2z_default.yaml`
- `tests/test_recipe_static_contracts.py`

Responsibilities:

- Implement a PyTorch Dataset that reads an a2z manifest and materializes one
  sample per row.
- Return the same core keys as `PAIDataset`.
- Call the official VLA preprocessor when `vla_preprocess_args` is provided.
- Preserve metadata keys such as `full_scene_id`, `source_dataset_dir`,
  `source_scene_name`, `scene_token`, `t0_timestamp_ns`, and residual summary.

Verification:

- Static config test that `sft_stage1_a2z_default.yaml` points to the new
  dataset target and the `default` VLA processor.
- Dataset unit test with synthetic image files and synthetic table fixtures.
- No-training collate smoke that verifies `tokenized_data` and `labels_mask`
  without model load.

### I4. Add no-training check command or script

Files likely touched:

- `scripts/check_a2z_sft_materialization.py`
- `tests/test_recipe_static_contracts.py`

Responsibilities:

- Load a tiny manifest.
- Instantiate `A2ZTrajectoryDataset` for train/val.
- Fetch N samples.
- Run collate.
- Print concise evidence: row keys, tensor shapes, tokenized fields, label mask
  counts, split validation result, and forbidden-label check result.
- Exit nonzero on missing keys, split leakage, residual hard fail, processor
  failure, or collate failure.

Verification:

- CLI argument parse/static test.
- Synthetic fixture run in CPU tests if it does not require heavy VLM processor
  downloads. If the real processor is too heavy for CI, keep CI on validator and
  dataset shape tests, and document the local venv command separately.

### I5. Add user learning/review note

Files likely touched:

- `docs/2026-06-17 [Note] Track C a2z SFT Training Structure Learning Guide.md`

Responsibilities:

- Explain Stage 1/default, Stage 1/nav, Stage 2, and evaluation in user-facing
  language.
- Include diagrams or ordered flows for data and loss.
- Include review questions, answer key, and wrong-answer signs.
- Explicitly state what cannot be claimed after materialization or micro smoke.

Verification:

- Self-check that every critical training-start decision has a review question.
- User review before any training approval.

## Acceptance Criteria

Feasibility acceptance:

- The user can review a written explanation of SFT structure and training
  principles.
- The Plan separates evidence, inference, unknowns, and forbidden claims.
- The first target is justified as Stage 1/default trajectory unless a route
  labeling policy is later approved.
- No training or checkpoint creation has occurred.

Implementation-readiness acceptance:

- A tiny a2z manifest can be validated without touching source data.
- An a2z dataset item can produce the official sample dict contract.
- The official VLA preprocessor can produce `tokenized_data` for a2z rows.
- The official collator can produce `labels_mask` for `traj_future`.
- Split validation blocks log/scene leakage.
- `pred_*` cannot enter any label path.
- The no-training check command has explicit failure signals.

Training-start acceptance, later and separate:

- User understanding gate passed.
- Feasibility implementation reviewed.
- Checkpoint base path and conversion state known.
- GPU/runtime envelope approved.
- Scratch/output path and retention limit approved.
- External logging policy approved.
- Micro SFT smoke command reviewed before execution.

## Risks and Mitigations

Risk: treating Track B predictions as labels.

Mitigation: validator rejects `pred_*` label fields; documentation states source
`ego_future_*` is the only trajectory supervision source.

Risk: train/val leakage through adjacent windows or same log.

Mitigation: log-level split for real SFT; `full_scene_id` cannot cross splits;
scene split only allowed for non-generalization loader smoke.

Risk: route-conditioned SFT without route labels.

Mitigation: first target is default trajectory SFT; `nav` requires a separate
approved route-label policy.

Risk: heavy processor/model behavior hidden by static tests.

Mitigation: separate CPU unit tests from local no-training processor/collate
smoke; do not claim readiness until the latter passes.

Risk: local hardware cannot support official 8-GPU recipe assumptions.

Mitigation: micro smoke is capped and classified as wiring/OOM discovery unless
run on an approved training host.

Risk: path and timestamp quirks in a2z records.

Mitigation: normalize filename separators and parse timestamp strings as
integers at the table-loader boundary.

## Open Decisions

1. Whether Stage 1/default trajectory is accepted as the first a2z SFT target.
2. Whether implementation should create a new `sft_stage1_a2z_default.yaml` or
   use CLI overrides against `sft_base.yaml`.
3. Where tiny manifests and no-training check outputs should live:
   `docs/`, `logs/`, or a new lightweight `manifests/` directory.
4. Whether the first manifest should use Track B selected scenes, avoid them for
   later before/after comparison, or create a separate small loader-only set.
5. Which output/scratch root is allowed for any later micro SFT smoke.

## Review Questions for the User

1. In the default trajectory Stage 1 path, which fields are model context and
   which fields are supervised target?
2. Why is `pred_xyz` from Track B forbidden as a label?
3. Why is log-level split stronger than window-level split?
4. What does the no-training materialization check prove, and what does it not
   prove?
5. Why should Stage 2 wait until after Stage 1/checkpoint policy review?
6. What additional policy is required before using the `nav` processor on a2z?

## Answer Key

1. Context: images, camera indices, relative timestamps, prompt, and ego
   history. Target: future trajectory through `ego_future_*`, future trajectory
   tokens, and label mask.
2. `pred_xyz` is the model's pretrained output, not source ground truth. Using
   it would imitate the current model rather than learn from a2z ego motion.
3. Adjacent windows and scenes from the same log can share nearly identical
   visual, route, and motion context. Log-level split reduces that leakage.
4. It proves adapter/processor/collator compatibility and failure signaling. It
   does not prove learning, model improvement, safety, or generalization.
5. Stage 2 consumes a Stage 1 VLM checkpoint and trains the action expert. It is
   higher runtime/checkpoint risk and should not be opened before Stage 1 data
   contract and checkpoint policy are clear.
6. a2z needs an approved route-text labeling policy. Without it, use the
   default trajectory processor rather than inventing `nav_text`.

## Completion State

This Plan is approved, active, and canonical. The corresponding Root Task now
tracks execution state, user understanding review, implementation subtasks, and
any later training-start decision gate. Keep this Plan's approved body fixed as
the execution baseline.

## Admin Changelog

- 2026-06-17 14:49:50 KST: status/canonical/approved_at/root_task updated after user approval.
