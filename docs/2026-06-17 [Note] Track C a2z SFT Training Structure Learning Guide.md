---
doc_type: note
status: review
topic: track-c-a2z-sft-training-structure-learning-guide
parent_plan: docs/2026-06-17 [Plan] Track C a2z SFT Feasibility and Implementation Readiness.md
parent_task: docs/2026-06-17 [Task] Track C a2z SFT Readiness Root Task.md
subtask: docs/2026-06-17 [Task] Track C F0 SFT Training Structure Learning Review.md
created_at: 2026-06-17 14:49:50 KST
updated_at: 2026-06-17 14:49:50 KST
---

# Track C a2z SFT Training Structure Learning Guide

## Document Frame

Purpose: make the Alpamayo 1.5 SFT structure understandable enough that the
user can review the a2z training plan before any training run is authorized.

Primary reader: the user deciding whether Track C may move from feasibility
planning into a2z SFT implementation.

Decision question: what exactly is being trained, which data fields are model
inputs versus supervised targets, and why must a2z materialization be validated
before any micro SFT run?

Exclusion scope: this Note does not approve training, create checkpoints, load
model weights, mutate a2z source data, define final hyperparameters, or claim
model improvement.

## One-Sentence Mental Model

SFT teaches Alpamayo to produce the expected future trajectory tokens from past
visual context and ego-motion history; for a2z, the future trajectory target
must come from source ego poses, not from Track B model predictions.

## The Recipe Execution Path

The Alpamayo 1.5 SFT recipe is configured by Hydra and executed through a
single training entry point:

```text
recipes/alpamayo1_5_sft/train_hf.py
  -> instantiate cfg.model
  -> instantiate cfg.data.train_dataset
  -> instantiate cfg.data.val_dataset
  -> instantiate cfg.data.collate_fn
  -> ReasoningVLA_Trainer(...)
  -> trainer.train()
```

The important consequence is that a2z does not need a new trainer first. It
needs a dataset/materialization path that returns the same semantic sample
fields expected by the existing processor, collator, and model.

## Stage 1/default Trajectory SFT

Stage 1/default trajectory is the recommended first a2z target.

Config shape:

```text
vla_processor/default.yaml
  components_order: image, traj_history, prompt, traj_future
  label_components: traj_future
```

What the model sees as context:

- camera history images;
- camera indices;
- relative timestamps;
- ego history trajectory tokens;
- a prompt asking for the future trajectory.

What is supervised:

- future trajectory tokens derived from `ego_future_xyz` and `ego_future_rot`;
- only the `traj_future` component is label-masked for loss.

Why this is the right first a2z target:

- it does not require route text;
- it directly matches the trajectory data Track B already proved can be rebuilt
  from a2z source records;
- it exercises the official processor/collator/label-mask path without adding a
  new labeling policy.

## Stage 1/navigation SFT

Navigation SFT is similar to default trajectory SFT, but it adds route
conditioning:

```text
vla_processor/nav.yaml
  components_order: image, traj_history, route, prompt, traj_future
  label_components: traj_future
```

The target is still future trajectory. The difference is that the model is also
conditioned on `nav_text`.

Why this is not the first a2z target:

- a2z currently provides ego motion and sensor records, not approved route text;
- inventing `nav_text` would create a new label policy and possible false
  supervision;
- route labeling needs a separate decision before it can enter training.

Safe conclusion: use default trajectory SFT first. Revisit navigation SFT only
after a route-text source and labeling policy are approved.

## Stage 2 Action-Expert SFT

Stage 2 is a later gate. It trains the trajectory/action expert after a Stage 1
trajectory checkpoint exists.

Stage 2 uses:

- base converted Alpamayo checkpoint as the model structure;
- Stage 1 VLM checkpoint as `stage1_vlm_checkpoint_path`;
- `ego_history_*` and `ego_future_*` to construct action-space diffusion
  training data;
- no DeepSpeed in the shipped Stage 2 nav config.

Why it should wait:

- it depends on a Stage 1 checkpoint policy;
- it creates higher checkpoint/storage/runtime risk;
- it is harder to interpret if the dataset contract is not already proven;
- it is unnecessary for the first no-training a2z readiness proof.

## Evaluation Path

Evaluation uses the validation dataset, collator, model sampling, and trajectory
metrics:

```text
evaluate_hf.py
  -> val_dataset
  -> collate_fn
  -> ReasoningSampler
  -> DistanceMetrics
```

Evaluation compares model-generated `pred_xyz` and `pred_rot` against
`ego_future_xyz` and `ego_future_rot`.

Important boundary:

- `pred_*` is output;
- `ego_future_*` is ground truth;
- output must not become the label for later supervised training.

## a2z Sample Contract

For Stage 1/default trajectory SFT, one a2z row should become one sample dict
with these core fields:

| Field | Role | Source |
| --- | --- | --- |
| `image_frames` | input context | a2z `samples/<camera>/...jpg` |
| `camera_indices` | input context | reviewed front-three mapping `[0, 1, 2]` |
| `relative_timestamps` | input context | camera frame offsets around `t0` |
| `ego_history_xyz` | input context | source `ego_pose` transformed into `t0` frame |
| `ego_history_rot` | input context | source `ego_pose` transformed into `t0` frame |
| `ego_future_xyz` | supervised target / eval GT | source `ego_pose` after `t0` |
| `ego_future_rot` | supervised target / eval GT | source `ego_pose` after `t0` |
| `tokenized_data` | processor output | official VLA preprocessor |
| `labels_mask` | collator output | official label-mask logic |

Metadata should also travel with the sample for review:

- `source_dataset_dir`;
- `source_scene_name`;
- `scene_token`;
- `full_scene_id`;
- `t0_sample_index`;
- `t0_timestamp_ns`;
- `camera_set`;
- residual summary.

## What the Processor Does

The VLA processor is the bridge between raw tensors and language-model-style
training input.

It performs these steps:

1. Sort images by camera index.
2. Build a chat-template conversation from components such as image,
   trajectory history, prompt, and future trajectory.
3. Convert images through the Qwen image processor.
4. Expand image placeholders into the correct number of image tokens.
5. Return `tokenized_data` containing text and image-processor outputs.

The dataset should not hand-build final model tokens. It should provide the raw
sample fields and call the official processor.

## What the Collator Does

The collator batches processed samples.

It performs these steps:

1. Stack tensor fields where possible.
2. Keep raw `image_frames` unstacked because image sizes may vary.
3. Tokenize the processed conversation text with padding.
4. Concatenate image processor outputs.
5. Build `labels_mask` for configured label components.

For default trajectory SFT, `labels_mask` should supervise `traj_future`.
History and image context should not be treated as prediction targets.

## What the Model Loss Means

In Stage 1/default trajectory SFT:

- `ego_history_*` helps create history trajectory tokens in the prompt;
- `ego_future_*` becomes future trajectory tokens;
- `labels_mask` determines which token positions contribute to supervised
  loss;
- the loss means "how well the model predicts the supervised future trajectory
  token component under this prompt."

This is not a direct safety metric. It is not closed-loop performance. It is
not proof that driving improved.

In Stage 2:

- `ego_future_*` is converted into action-space diffusion training data;
- the action expert learns against that trajectory/action representation;
- this should only happen after Stage 1 and checkpoint policy are reviewed.

## Why Track B Outputs Are Not Labels

Track B is pretrained inference evidence. It produces `pred_xyz` and `pred_rot`
from the existing model.

Those predictions are useful for:

- before/after qualitative comparison;
- evaluation baselines;
- scene selection context;
- debugging runtime behavior.

They are forbidden as SFT labels because using them would train the model to
imitate itself. The supervised label must come from the source a2z ego-motion
record, specifically the future ego poses transformed into the selected `t0`
local frame.

## Why Splits Matter

a2z rows are highly correlated:

- neighboring windows share frames;
- neighboring windows share ego history and future trajectory;
- scenes in the same log share acquisition conditions and route context.

A window-level split can put nearly identical data into both train and val. That
would make validation look better without proving generalization.

Default policy:

- use `source_dataset_dir` as the real split boundary;
- never allow the same `full_scene_id` to cross splits;
- permit scene-level split only for loader/debug smoke and label it as
  non-generalization evidence.

## No-Training Materialization Check

This is the first real proof before training.

It should do:

- read a tiny a2z manifest;
- materialize 20-40 rows from source records;
- load front-three image history from source paths;
- rebuild history/future ego tensors from source `ego_pose`;
- call the official VLA preprocessor;
- run the official collator;
- report shapes, tokenized fields, label-mask counts, split status, and residual
  status.

It should not do:

- load model weights;
- call `trainer.train()`;
- create checkpoints;
- copy the source dataset;
- write into the source a2z dataset;
- use Track B `pred_*` as labels.

Pass meaning:

- "The a2z data path can feed the official SFT processor/collator."

Non-meaning:

- "Training will improve the model."
- "The model is safer."
- "The split proves generalization."
- "The runtime envelope is approved."

## Later Micro SFT Smoke

A micro SFT smoke is a later, separately approved action.

Narrow proposal:

- Stage 1/default trajectory first;
- 20 train rows and 5-10 validation rows;
- rows from different source logs where possible;
- strict step cap;
- external logging disabled unless approved;
- `save_total_limit: 1`;
- output under an approved scratch root;
- first local run classified as wiring/OOM discovery if hardware is below the
  recipe's official 8-GPU validation envelope.

Stop before launch if any of these is unclear:

- user understanding review;
- checkpoint conversion path;
- GPU/runtime envelope;
- scratch/output path;
- checkpoint retention limit;
- external logging policy;
- exact command line.

## Review Questions

1. In Stage 1/default trajectory SFT, what fields are input context?
2. In Stage 1/default trajectory SFT, what fields are supervised target?
3. What does `labels_mask` do?
4. Why should a2z use default trajectory before nav?
5. Why is `pred_xyz` forbidden as a label?
6. Why is log-level split safer than window-level split?
7. What does the no-training materialization check prove?
8. What does a micro SFT smoke not prove?
9. Why should Stage 2 wait?
10. What must be approved immediately before any training command runs?

## Answer Key

1. Input context: images, camera indices, relative timestamps, ego history, and
   prompt text.
2. Supervised target: future trajectory, represented through `ego_future_xyz`,
   `ego_future_rot`, future trajectory tokens, and the `traj_future` label
   component.
3. `labels_mask` selects which token positions contribute to supervised loss.
   For default trajectory SFT, it should supervise `traj_future`, not image or
   history context.
4. Default trajectory does not need route text. Nav requires `nav_text`, and
   a2z does not yet have an approved route-label source.
5. `pred_xyz` is model output from pretrained inference. Using it as a label
   would imitate the current model instead of learning from source ego motion.
6. Window-level split leaks nearly identical frames and trajectories across
   train/val. Log-level split reduces that leakage.
7. It proves dataset, processor, collator, label-mask, split validation, and
   failure signaling can work without model training.
8. It does not prove model improvement, safety, generalization, or closed-loop
   performance.
9. Stage 2 depends on a Stage 1 checkpoint and trains the action expert, so it
   has higher runtime and checkpoint risk.
10. User understanding, checkpoint paths, GPU/runtime envelope, scratch/output
    path, retention limit, logging policy, and exact command must be approved.

## Wrong-Answer Signs

- Saying `ego_future_*` is a normal model input rather than supervision/GT.
- Treating Track B `pred_*` as acceptable training labels.
- Calling a no-training materialization check a training result.
- Treating a tiny train/val split as generalization evidence.
- Starting with navigation SFT while route labels are still undefined.
- Treating Stage 2 as required for the first a2z readiness proof.
- Equating lower training loss with safety or closed-loop driving quality.

## Review Verdict Template

Use this before moving to implementation:

```text
SFT structure review:
- Stage 1/default trajectory understood: yes/no
- Input vs label boundary understood: yes/no
- Track B pred_* label ban understood: yes/no
- Split/leakage policy understood: yes/no
- No-training check meaning understood: yes/no
- Stage 2 deferral understood: yes/no
- Remaining questions:
- Verdict: proceed_to_implementation / revise_learning_guide / stop
```

## Current Status

This Note is ready for user review. It is a learning artifact only. It does not
authorize implementation, training, checkpoint creation, or model-weight loading.
