---
doc_type: plan
status: active
plan_id: pai-nav-annotation-inventory
version: 1
canonical: true
created_at: 2026-06-17 18:29:56 KST
approved_at: 2026-06-17 18:29:56 KST
root_plan:
supersedes:
superseded_by:
root_task: docs/2026-06-17 [Task] PAI Dataset Nav Annotation Inventory Root Task.md
revision_type: initial_inventory_plan
revision_reason: Establish a read-only PAI dataset and nav annotation inventory baseline before any Alpamayo 1.5 SFT preparation
source_repo: /home/user/Workspace/alpamayo-recipes
related_inference_repo: /home/user/Workspace/alpamayo1.5
related_historical_repo: /home/user/Workspace/alpamayo
---

# PAI Dataset Nav Annotation Inventory Plan

## Document Frame

Purpose: define the approved read-only workflow for checking local PAI dataset
state, nav annotation availability, and user-run download/status tooling needed
before Alpamayo 1.5 SFT/nav training can be considered.

Primary reader: the user deciding which PAI dataset root is canonical, which
components to download next, and how nav annotations should be sourced.

Decision question: what is currently present locally, what is missing for the
Alpamayo 1.5 nav recipe, and what can safely be downloaded or monitored by the
user without mutating source data or starting training?

Exclusion scope: this Plan does not approve large downloads, SFT/training,
model checkpoint creation, source dataset mutation, git commit, git push, or
synthetic training-ready nav labels.

## Current Working Assumptions

- Candidate PAI roots must be discovered from repo docs, environment, and local
  filesystem evidence, then reported separately.
- `/data/datasets/physical_ai_av` is memory-derived historical evidence and
  must be re-verified in this run before being treated as current.
- `/mnt/zfs_pool/physical_ai_av` is a separate local candidate and must not be
  merged with `/data/datasets/physical_ai_av` unless the filesystem proves they
  are the same path.
- `nav_demo_samples.json` from the Alpamayo 1.5 repo is an official smoke-test
  annotation source if present locally, but it only proves annotation coverage
  for its listed samples.
- Future trajectory-derived route text is leakage-prone and cannot be promoted
  to official navigation ground truth.

## Evidence Baseline To Gather

- Repo instructions and recipe docs:
  `README.md`, `recipes/alpamayo1_5_sft/README.md`,
  `recipes/alpamayo1_5_sft/SKILL.md`.
- Downloader and dataset code:
  `scripts/download_pai.py`, `src/alpamayo/data/pai.py`,
  `src/alpamayo/data/pai_nav.py`, `src/alpamayo/data/pai_utils.py`.
- Local metadata:
  `features.csv`, `clip_index.parquet`, `metadata/feature_presence.parquet`,
  component files, reasoning parquet/json files, and nav annotation JSON/parquet
  files under candidate roots and related repos.

## Work Plan

### P0. Root Candidate Discovery

Goal: identify all plausible local PAI dataset roots and keep their status
separate.

Actions:

- Search repo docs/configs, environment variables, and known local dataset
  paths for PAI roots.
- Confirm candidate root validity by checking at least `features.csv`,
  `clip_index.parquet`, and `metadata/feature_presence.parquet`.
- Record whether candidates are identical paths, symlinks, or independent
  directories.

Verification:

- A root summary table names each candidate and its required metadata presence.

### P1. Clip Catalog Inventory

Goal: extract the PAI metadata clip catalog without loading training data.

Actions:

- Read `clip_index.parquet` for each valid candidate root.
- Preserve machine-readable clip lists with `clip_id`, `chunk`, and available
  keyframe/t0-related columns when present.
- Keep large catalogs out of chat and store them under `logs/`.

Verification:

- A machine-readable clip list exists per valid candidate root, or the report
  explains why it could not be produced.

### P2. Component Inventory

Goal: distinguish downloaded component files, metadata-present but missing
files, and metadata-absent components.

Actions:

- Parse `features.csv` and `metadata/feature_presence.parquet`.
- Check camera, calibration, egomotion labels, `features.csv`,
  `clip_index.parquet`, `metadata/feature_presence.parquet`, and
  reasoning/nav parquet or JSON files.
- Produce CSV/JSON/Markdown inventory artifacts under `logs/` and/or `docs/`.

Verification:

- Each required component status is classified as `downloaded`,
  `metadata_present_file_missing`, or `metadata_absent`.

### P3. Nav Annotation Gap Analysis

Goal: report whether code-required nav annotations exist and join to the local
PAI metadata.

Actions:

- Treat `PAIDatasetWithNav` code as the authoritative schema:
  `clip_id`, `t0_relative`, and `nav_text`.
- Search current repo, related repos, and candidate dataset roots for
  `nav_demo_samples.json` or similar nav annotation files.
- Join official/local annotation rows to each candidate root by `clip_id` and
  chunk when possible.

Verification:

- The report clearly states official source found/not found, sample counts,
  join coverage, and whether annotations are sufficient only for loader smoke
  or for broader SFT training.

### P4. User-Run Download And Status Tooling

Goal: provide user-executable scripts without starting large downloads.

Actions:

- Reuse or wrap `scripts/download_pai.py`; do not rewrite a downloader.
- Add dry-run/plan support, skip/resume via Hugging Face snapshot behavior,
  log path, and failed/missing summary capture.
- Add a read-only status script that recalculates component completion from
  local metadata and filesystem state.

Verification:

- Dry-run/status commands complete locally without downloading large payloads.
- User commands are documented with exact paths.

### P5. Final Requirements Review

Goal: prevent overclaiming training readiness.

Actions:

- Map each user requirement to evidence, code path, artifact, or unknown.
- Separate Evidence, Inference, and Unknown.
- State current proven facts, unproven facts, and next decisions.

Completion condition:

- The user has a report, clip catalog path, inventory artifact, nav gap
  analysis, dry-run download command, status monitoring command, and decision
  list.
