---
doc_type: report
status: needs_review
created_at: 2026-06-22 11:28:00 KST
source_repo: /home/user/Workspace/alpamayo-recipes
plan: docs/2026-06-22 [Plan] Track C PAI SFT Demo Gate Before a2z.md
root_task: docs/2026-06-22 [Task] Track C PAI SFT Demo Root Task.md
execution_session_id: 019eed1d-b6bc-7562-880e-f198c3b5f67b
managing_session_id: 019ecfca-468c-7682-b75e-aadca21dfe86
---

# Track C PAI SFT Demo Result

## Current Conclusion

P0 preflight and P1 no-training batch proof passed for the official PAI nav
demo path.

This proves only that the 20-row / 19-unique-clip PAI nav demo can reach the
`PAIDatasetWithNav -> nav processor -> collator -> labels_mask` path without
model loading or training. It does not prove learning, Stage 1 trainability,
Stage 2 viability, broad PAI navigation training readiness, or a2z SFT
readiness.

Stage 1 must not start yet. The local HF cache contains a raw
`nvidia/Alpamayo-1.5-10B` snapshot, but the README-required converted A1-format
checkpoint path has not been verified.

## Evidence

### P0 Preflight

Command:

```bash
cd /home/user/Workspace/alpamayo-recipes/recipes/alpamayo1_5_sft
uv run python ../../scripts/check_pai_download_status.py \
  --roots /mnt/zfs_pool/physical_ai_av \
  --nav-annotations /home/user/Workspace/alpamayo1.5/notebooks/nav_demo_samples.json \
  --chunk-ids 214 224 276 317 420 727 728 968 982 1519 1657 1984 2277 2368 2372 2447 2599 2634 2868 \
  --output-dir /home/user/Workspace/alpamayo-recipes/logs/pai_sft_demo_preflight \
  --prefix 20260622_pai_sft_demo_p0
```

Output artifacts:

- `/home/user/Workspace/alpamayo-recipes/logs/pai_sft_demo_preflight/20260622_pai_sft_demo_p0__summary.md`
- `/home/user/Workspace/alpamayo-recipes/logs/pai_sft_demo_preflight/20260622_pai_sft_demo_p0__summary.json`
- `/home/user/Workspace/alpamayo-recipes/logs/pai_sft_demo_preflight/20260622_pai_sft_demo_p0__components_all_roots.csv`

Observed facts:

- PAI root metadata exists under `/mnt/zfs_pool/physical_ai_av`.
- Nav 19-chunk filtered catalog contains 1,877 clips, 19 chunks, and 36
  features.
- Required components are complete for all 19 checked chunks:
  `camera_front_wide_120fov`, `camera_cross_left_120fov`,
  `camera_cross_right_120fov`, `camera_front_tele_30fov`,
  `camera_intrinsics`, `sensor_extrinsics`, and `egomotion`.
- Nav annotation JSON has 20 rows and 19 unique clips.
- All 20 annotation rows match clips in the 19 configured chunks.
- Clip-level nav coverage is narrow: 19 clips with nav annotation and 1,858
  clips without nav annotation in the filtered catalog.
- GPU state: two RTX 4090 GPUs, each 24,564 MiB total memory, driver
  `590.48.01`.
- Recipe import passed for `alpamayo.data.pai_nav`,
  `alpamayo.processor.qwen_processor`, `alpamayo.chat_template.components`, and
  `alpamayo1_5_sft.train_hf`.
- DeepSpeed config exists at
  `/home/user/Workspace/alpamayo-recipes/recipes/alpamayo1_5_sft/configs/deepspeed/zero2.json`.
- `WANDB_MODE=disabled` was verified for the checked runtime shell.

Warnings:

- Import emitted a `pynvml` deprecation warning.
- Import emitted a `torchao` cpp-extension compatibility warning for
  `torch 2.8.0+cu128` and `torchao 0.15.0`.
- These warnings did not block imports or P1 batch proof.

Checkpoint state:

- A cached raw HF snapshot exists at
  `/home/user/.cache/huggingface/hub/models--nvidia--Alpamayo-1.5-10B/snapshots/f11cd25b758ab560114019b555dde2a8b92d88b4`.
- Its config reports `model_type: alpamayo1_5` and
  `architectures: ["Alpamayo1_5"]`.
- The README-required SFT checkpoint is a converted A1-format directory
  produced by `scripts/convert_checkpoint.py to-a1`.
- No verified converted A1-format checkpoint path was found during preflight.

### P1 No-Training Batch Proof

Command:

```bash
cd /home/user/Workspace/alpamayo-recipes/recipes/alpamayo1_5_sft
WANDB_MODE=disabled uv run python - <<'PY' 2>&1 | tee \
  /home/user/Workspace/alpamayo-recipes/logs/pai_sft_demo_preflight/20260622_pai_sft_demo_p1_batch_proof.log
# one-off proof script: instantiate PAIDatasetWithNav train/val,
# load one sample each, collate one batch each, check route text and labels_mask
PY
```

Log:

- `/home/user/Workspace/alpamayo-recipes/logs/pai_sft_demo_preflight/20260622_pai_sft_demo_p1_batch_proof.log`

Observed facts:

- Train dataset length: 20.
- Val dataset length: 20.
- Train sample `nav_text`: `Turn left in 11m`.
- Val sample `nav_text`: `Turn left in 11m`.
- Train and val samples both include `nav_text`, `tokenized_data`,
  `image_frames`, `camera_indices`, `relative_timestamps`, `ego_history_xyz`,
  `ego_future_xyz`, `label_components`, and `generation_mode`.
- Sample tensor shapes:
  - `image_frames`: `(4, 4, 3, 1080, 1920)`
  - `camera_indices`: `(4,)`
  - `relative_timestamps`: `(4, 4)`
  - `ego_history_xyz`: `(1, 16, 3)`
  - `ego_future_xyz`: `(1, 64, 3)`
- Train tokenized text contains route start token and the nav text.
- Val tokenized text contains route start token and the nav text.
- Train tokenized text contains future trajectory start token.
- Train `generation_mode`: `False`.
- Val `generation_mode`: `True`.
- `label_components`: `["traj_future"]`.
- Train batch `input_ids` shape: `(1, 3213)`.
- Val batch `input_ids` shape: `(1, 3082)`.
- Train `labels_mask` shape: `(1, 3213)`.
- Val `labels_mask` shape: `(1, 3082)`.

Mask proof:

- Train true mask count: 131.
- Future trajectory span count: 130.
- Assistant EOS mask count: 1.
- Unexpected train mask outside future span plus assistant EOS: 0.
- Missing train mask inside future span: 0.
- Route span masked token count: 0.
- Val true mask count: 0.
- Final proof marker: `P1_BATCH_PROOF_PASS`.

## Inference

- The official nav demo path is wired correctly through dataset loading,
  route-conditioned tokenized input construction, train/val collation, and
  `traj_future` label masking.
- `nav_text` is input-side route conditioning. It is not the supervised label.
- `traj_future` is the supervised label component for this Stage 1 nav path.
- The P1 pass is enough to request a Stage 1 bounded overfit smoke review, but
  it is not enough to start Stage 1 automatically.
- The raw cached Alpamayo 1.5 snapshot may be the input to conversion, but it
  should not be treated as the verified SFT checkpoint path until an A1-format
  output directory is created or supplied and checked.

## Unknown

- The converted A1-format checkpoint path is still unknown.
- Stage 1 local memory behavior on two RTX 4090 GPUs is unknown.
- Finite-loss behavior is unknown because no training was run.
- Stage 2/eval behavior is unknown because Stage 2/eval remains outside the
  approved scope.
- Broader PAI nav readiness is unknown because only 19 annotated clips are in
  scope.
- a2z SFT readiness is unknown because this gate did not touch a2z data.

## Requirement Review

| Requirement | Code path / artifact | Evidence |
| --- | --- | --- |
| Recheck PAI root/components/chunks | `scripts/check_pai_download_status.py` | 19/19 required chunks complete for all required components |
| Recheck nav JSON rows/clips | `nav_demo_samples.json` plus status script | 20 rows, 19 unique clips, 20 matched rows |
| Check GPU/env/DeepSpeed/W&B | shell checks and recipe imports | two RTX 4090 GPUs, import OK, DeepSpeed path exists, `WANDB_MODE=disabled` |
| Avoid model loading/training | P1 one-off script only instantiates dataset/processor/collator | no Trainer, no model instantiate, no checkpoint output |
| Verify route/nav text input | train/val tokenized text checks | route token and `Turn left in 11m` present |
| Verify label mask target | collated train batch mask comparison | mask equals `traj_future` span plus assistant EOS; route span unmasked |
| Verify val generation mask | collated val batch | all-false `labels_mask` |

## Non-Claims

- This is not PAI navigation training readiness.
- This is not a2z SFT readiness.
- This is not a generalization result.
- This is not model quality evidence.
- This is not Stage 2 readiness.
- This is not evidence that synthetic or future-derived route labels are valid.

## Stop Condition

Stop here before Stage 1.

Stage 1 requires both:

- explicit user approval for a bounded overfit smoke; and
- a verified A1-format checkpoint path for `model.checkpoint_path`.

Recommended first Stage 1 cap, if approved later:

```bash
trainer.max_steps=20
trainer.logging_steps=1
trainer.save_steps=20
trainer.save_total_limit=1
trainer.report_to=none
```

Stage 1 output must use a new non-colliding output directory and keep
`WANDB_MODE=disabled`.
