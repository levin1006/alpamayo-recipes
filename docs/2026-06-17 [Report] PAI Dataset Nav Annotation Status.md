---
doc_type: report
status: completed
created_at: 2026-06-17 18:38:00 KST
source_repo: /home/user/Workspace/alpamayo-recipes
related_inference_repo: /home/user/Workspace/alpamayo1.5
related_historical_repo: /home/user/Workspace/alpamayo
plan: docs/2026-06-17 [Plan] PAI Dataset Nav Annotation Inventory.md
root_task: docs/2026-06-17 [Task] PAI Dataset Nav Annotation Inventory Root Task.md
---

# PAI Dataset Nav Annotation Status Report

## Current Conclusion

Two local PAI dataset roots are present and should be treated separately:

- `/data/datasets/physical_ai_av`
- `/mnt/zfs_pool/physical_ai_av`

Both roots have the same global metadata catalog: `306,152` clips, `3,146`
chunks, and `36` feature/component definitions. They are not the same resolved
path.

For Alpamayo 1.5 nav smoke-test preparation, `/mnt/zfs_pool/physical_ai_av` is
the stronger current candidate because all 19 chunks referenced by the bundled
nav annotations have the required 4-camera, calibration, and egomotion payloads
downloaded. `/data/datasets/physical_ai_av` has only one of those 19 nav chunks
downloaded for the checked required components.

This does not prove full SFT training readiness. It proves local payload
coverage for the bundled 20-sample nav smoke set under the checked components.
It does not prove broader navigation-label coverage beyond that smoke set.

## Evidence

### Repo and code contract

- `recipes/alpamayo1_5_sft/README.md` says the nav recipe downloads the chunks
  referenced by nav annotations, with four cameras, `camera_intrinsics`,
  `sensor_extrinsics`, and `egomotion`.
- `recipes/alpamayo1_5_sft/configs/sft_stage1_nav.yaml` and
  `recipes/alpamayo1_5_sft/configs/sft_stage2_nav.yaml` configure
  `PAIDatasetWithNav` with the 19 nav chunks:
  `214 224 276 317 420 727 728 968 982 1519 1657 1984 2277 2368 2372 2447 2599 2634 2868`.
- `src/alpamayo/data/pai_nav.py` is the authoritative loader contract:
  each JSON row must provide `clip_id`, `t0_relative`, and `nav_text`. The
  loader uses `t0_relative` as `t0_us` and injects `nav_text` into the sample.
- `src/alpamayo/data/pai.py` and `src/alpamayo/data/pai_utils.py` load
  `features.csv`, `clip_index.parquet`, `metadata/feature_presence.parquet`,
  and feature chunk files. Reasoning parquet is optional and separate from
  nav annotation JSON.
- `scripts/download_pai.py` already provides the real Hugging Face snapshot
  download path, so the new download helper only wraps it.

### Root discovery

Environment variables did not define a PAI root. Filesystem checks found:

| Root | Required metadata | Clips | Chunks | Features |
| --- | --- | ---: | ---: | ---: |
| `/data/datasets/physical_ai_av` | present | 306,152 | 3,146 | 36 |
| `/mnt/zfs_pool/physical_ai_av` | present | 306,152 | 3,146 | 36 |

Required metadata files present in both:

- `features.csv`
- `clip_index.parquet`
- `metadata/feature_presence.parquet`
- `metadata/data_collection.parquet`

Neither root has `reasoning/ood_reasoning.parquet`.

### Full clip catalog

The full clip catalog is large, so it is stored as CSV:

- `/home/user/Workspace/alpamayo-recipes/logs/pai_inventory/20260617_pai_nav_inventory__data__datasets__physical_ai_av__clips.csv`
- `/home/user/Workspace/alpamayo-recipes/logs/pai_inventory/20260617_pai_nav_inventory__mnt__zfs_pool__physical_ai_av__clips.csv`

Each catalog includes available metadata columns from `clip_index.parquet`:
`clip_id`, `chunk`, `split`, and `clip_is_valid`.

### Component inventory

Full component inventory:

- `/home/user/Workspace/alpamayo-recipes/logs/pai_inventory/20260617_pai_nav_inventory__components_all_roots.csv`
- `/home/user/Workspace/alpamayo-recipes/logs/pai_inventory/20260617_pai_nav_inventory__summary.md`
- `/home/user/Workspace/alpamayo-recipes/logs/pai_inventory/20260617_pai_nav_inventory__summary.json`

Nav 19-chunk subset inventory:

- `/home/user/Workspace/alpamayo-recipes/logs/pai_inventory/20260617_pai_nav_chunks_inventory__components_all_roots.csv`
- `/home/user/Workspace/alpamayo-recipes/logs/pai_inventory/20260617_pai_nav_chunks_inventory__summary.md`
- `/home/user/Workspace/alpamayo-recipes/logs/pai_inventory/20260617_pai_nav_chunks_inventory__summary.json`

Nav annotation coverage inventory:

- `/home/user/Workspace/alpamayo-recipes/logs/pai_inventory/20260617_nav_annotation_coverage__mnt__zfs_pool__physical_ai_av__clips.csv`
- `/home/user/Workspace/alpamayo-recipes/logs/pai_inventory/20260617_nav_annotation_coverage__mnt__zfs_pool__physical_ai_av__clips_with_nav_annotations.csv`
- `/home/user/Workspace/alpamayo-recipes/logs/pai_inventory/20260617_nav_annotation_coverage__mnt__zfs_pool__physical_ai_av__clips_without_nav_annotations.csv`
- `/home/user/Workspace/alpamayo-recipes/logs/pai_inventory/20260617_nav_annotation_coverage__summary.md`

Nav 19-chunk required-component summary:

| Root | Required components | Downloaded chunks per component | Missing chunks per component |
| --- | --- | ---: | ---: |
| `/data/datasets/physical_ai_av` | 4 cameras, `camera_intrinsics`, `sensor_extrinsics`, `egomotion` | 1 / 19 | 18 / 19 |
| `/mnt/zfs_pool/physical_ai_av` | 4 cameras, `camera_intrinsics`, `sensor_extrinsics`, `egomotion` | 19 / 19 | 0 / 19 |

Classification rule:

- `downloaded`: expected chunk file exists.
- `metadata_present_file_missing`: component exists in `features.csv` and
  `feature_presence.parquet`, but the expected chunk file is absent.
- `metadata_absent`: component is absent from metadata. This was `0` for the
  checked nav components in both roots.

## Nav Annotation Gap Analysis

Found official/local nav smoke annotation:

- `/home/user/Workspace/alpamayo1.5/notebooks/nav_demo_samples.json`

Schema:

- rows: `20`
- columns: `clip_id`, `t0`, `clip_start_timestamp`, `t0_relative`,
  `nav_text`, `nav_maneuver`, `distance_m`, `cot`
- code-required fields present: `clip_id`, `t0_relative`, `nav_text`

Join coverage:

| Root | Annotation rows | Matched by `clip_id` | Matched chunks |
| --- | ---: | ---: | --- |
| `/data/datasets/physical_ai_av` | 20 | 20 | 19 nav chunks |
| `/mnt/zfs_pool/physical_ai_av` | 20 | 20 | 19 nav chunks |

Interpretation:

- The annotation file is enough for the documented Alpamayo 1.5 20-sample
  overfit/loader smoke path.
- It is not evidence of broad nav SFT label coverage across the full PAI
  catalog.
- The 20 annotation rows cover 19 unique clips. One clip has two annotated
  route samples.
- Clip catalogs generated by `scripts/check_pai_download_status.py` now include
  `nav_annotation_count` and `has_nav_annotation`. The script also writes
  separate `clips_with_nav_annotations.csv` and
  `clips_without_nav_annotations.csv` files.
- For the `/mnt/zfs_pool/physical_ai_av` nav 19-chunk subset, only 19 of 1,877
  clips have nav annotations. The other 1,858 clips should not be forced into
  nav SFT unless a real nav annotation source is supplied.
- No broader official nav annotation parquet/json was found in the searched
  current repo, related repos, or candidate dataset roots.
- `reasoning/ood_reasoning.parquet` is absent from both candidate roots and is
  not a substitute for `PAIDatasetWithNav` route text.

## Nav Annotation Fill Options

Allowed for real SFT:

- Official nav annotation source that provides `clip_id`, timestamp/keyframe
  alignment, and route/navigation text from an approved source.
- Human-authored annotation with reviewable guidelines and no future-label
  leakage.
- Map/route-planner generated route text if it is generated from permissible
  map/planner inputs available at or before the decision time, with auditable
  alignment to `clip_id` and `t0_relative`.

Allowed only for loader smoke:

- Placeholder route text, clearly marked as synthetic and never used as
  training/evaluation ground truth.

Forbidden as training GT by default:

- Route text reverse-engineered from future trajectory. That can leak the
  target future path into the conditioning signal and must not be called
  official navigation ground truth.

## User-Run Download Command

Dry-run plan, no download:

```bash
cd /home/user/Workspace/alpamayo-recipes/recipes/alpamayo1_5_sft
uv run python ../../scripts/plan_pai_nav_download.py \
  --output-dir /mnt/zfs_pool/physical_ai_av \
  --logs-dir /home/user/Workspace/alpamayo-recipes/logs/pai_download
```

Actual download, user-run only:

```bash
cd /home/user/Workspace/alpamayo-recipes/recipes/alpamayo1_5_sft
export HF_TOKEN=<your Hugging Face token>
uv run python ../../scripts/plan_pai_nav_download.py \
  --output-dir /mnt/zfs_pool/physical_ai_av \
  --logs-dir /home/user/Workspace/alpamayo-recipes/logs/pai_download \
  --execute
```

The wrapper calls `scripts/download_pai.py` with:

- chunks: `214 224 276 317 420 727 728 968 982 1519 1657 1984 2277 2368 2372 2447 2599 2634 2868`
- cameras: `camera_front_wide_120fov`, `camera_cross_left_120fov`,
  `camera_cross_right_120fov`, `camera_front_tele_30fov`
- calibration: `camera_intrinsics`, `sensor_extrinsics`
- labels: `egomotion`

Hugging Face `snapshot_download` handles existing local files as skip/resume
inputs. The wrapper records the command log and writes a failure JSON if the
wrapped downloader exits non-zero.

## Monitoring Commands

Tail download log:

```bash
cd /home/user/Workspace/alpamayo-recipes
tail -f logs/pai_download/pai_nav_download_*.log
```

Recalculate full local status:

```bash
cd /home/user/Workspace/alpamayo-recipes/recipes/alpamayo1_5_sft
uv run python ../../scripts/check_pai_download_status.py \
  --roots /data/datasets/physical_ai_av /mnt/zfs_pool/physical_ai_av \
  --nav-annotations /home/user/Workspace/alpamayo1.5/notebooks/nav_demo_samples.json \
  --output-dir /home/user/Workspace/alpamayo-recipes/logs/pai_inventory
```

Recalculate only nav 19-chunk status:

```bash
cd /home/user/Workspace/alpamayo-recipes/recipes/alpamayo1_5_sft
uv run python ../../scripts/check_pai_download_status.py \
  --roots /data/datasets/physical_ai_av /mnt/zfs_pool/physical_ai_av \
  --nav-annotations /home/user/Workspace/alpamayo1.5/notebooks/nav_demo_samples.json \
  --chunk-ids 214 224 276 317 420 727 728 968 982 1519 1657 1984 2277 2368 2372 2447 2599 2634 2868 \
  --output-dir /home/user/Workspace/alpamayo-recipes/logs/pai_inventory
```

## Inference

- `/mnt/zfs_pool/physical_ai_av` is the user-approved canonical candidate for
  Alpamayo 1.5 nav smoke.
- `/data/datasets/physical_ai_av` appears to be a smaller selected-root or
  older partial root for the checked nav components.
- The bundled `nav_demo_samples.json` supports loader/overfit smoke only; it
  should not be described as full nav SFT coverage.
- Until a broader real nav source is available, SFT/nav training candidate
  lists should be restricted to rows with `has_nav_annotation=True`.
- Clips without nav annotation should be reserved for other tasks, such as
  default trajectory SFT, component smoke, retrieval/inventory checks, or future
  annotation work.

## Unknown

- Whether a broader official nav annotation source exists outside the searched
  local repos/dataset roots.
- Whether placeholder route text should be permitted for loader smoke only.
- Whether additional components beyond the checked nav minimum should be
  downloaded next, such as offline calibration, vehicle dimensions, lidar,
  radar, obstacle labels, or reasoning parquet.

## Current Proof Boundary

Proven:

- Both candidate roots have the full global PAI metadata catalog.
- `/mnt/zfs_pool/physical_ai_av` has the checked required component files for
  all 19 nav annotation chunks.
- The local `nav_demo_samples.json` has the schema required by
  `PAIDatasetWithNav` and all 20 rows join to `clip_index.parquet`.
- No broad nav annotation file beyond the 20-sample demo JSON was found in the
  searched local locations.

Not proven:

- Full PAI component completeness.
- Full navigation SFT readiness.
- Official broad nav annotation availability.
- Training correctness or model quality.

## Requirements Review

| Requirement | Evidence path | Review result |
| --- | --- | --- |
| PAI root candidates separated | this report, root table; `logs/pai_inventory/*summary*` | satisfied |
| Full clip list available | per-root `*__clips.csv` files | satisfied |
| Component inventory with status classes | `scripts/check_pai_download_status.py`; `*__components*.csv` | satisfied |
| Nav schema and file search | `src/alpamayo/data/pai_nav.py`; local `nav_demo_samples.json` join summary | satisfied |
| Nav fill policy without leakage | Nav Annotation Fill Options section | satisfied |
| Nav-missing clips reflected in file lists | `clips_without_nav_annotations.csv`; `has_nav_annotation` in clip catalog | satisfied |
| User-run download/sync script | `scripts/plan_pai_nav_download.py` dry-run verified | satisfied |
| Monitoring/status script | `scripts/check_pai_download_status.py` full and nav-subset runs verified | satisfied |
| No prohibited runtime work | no large download, no training, no checkpoint, no dataset mutation, no commit/push | satisfied |

## Next Decisions

1. Select canonical PAI root: recommended candidate is
   `/mnt/zfs_pool/physical_ai_av`. User approved.
2. Decide whether to download more than the current nav minimum components.
3. Decide how to obtain broader nav annotation source: official file, human
   annotation, or planner/map-derived labels.
4. Keep non-annotated clips out of nav SFT unless a real annotation source is
   added.
5. Decide whether placeholder route text is allowed strictly for loader smoke.
