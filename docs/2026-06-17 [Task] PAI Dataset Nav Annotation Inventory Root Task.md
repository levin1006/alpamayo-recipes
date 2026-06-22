---
doc_type: task
task_role: root
status: needs_review
task_id: pai-nav-annotation-inventory-root
parent_plan: docs/2026-06-17 [Plan] PAI Dataset Nav Annotation Inventory.md
parent_task:
created_at: 2026-06-17 18:29:56 KST
updated_at: 2026-06-17 18:38:00 KST
---

# PAI Dataset Nav Annotation Inventory Root Task

## Purpose

Track execution against the approved PAI Dataset Nav Annotation Inventory Plan.
This task does not authorize large downloads, SFT/training, model checkpoint
creation, source dataset mutation, git commit, or git push.

## Canonical Plan

- `docs/2026-06-17 [Plan] PAI Dataset Nav Annotation Inventory.md`

## Current Roll-Up

Status: `needs_review`

Read-only discovery and tooling preparation are complete. No large download,
training, checkpoint creation, source dataset mutation, git commit, or git push
was executed. The final report now needs user review for canonical root and nav
annotation policy decisions.

## Required Subtask Registry

| Subtask | Status | First action |
| --- | --- | --- |
| P0 root candidate discovery | done | Verified metadata presence under each candidate root |
| P1 clip catalog inventory | done | Exported per-root clip catalog from `clip_index.parquet` |
| P2 component inventory | done | Classified component files by metadata and filesystem state |
| P3 nav annotation gap analysis | done | Validated `nav_demo_samples.json` schema and clip join coverage |
| P4 user-run tooling | done | Added dry-run download wrapper/status script |
| P5 requirements review | needs_review | User reviews evidence, decisions, and proof boundary |

## Blockers

- Canonical dataset root is not yet chosen by the user.
- Full SFT training readiness cannot be claimed until broader nav annotation
  source and component completeness policy are proven.

## Next Decision

Review `docs/2026-06-17 [Report] PAI Dataset Nav Annotation Status.md`, then decide:

- which dataset root should be canonical;
- which component set to download next;
- which official or human-approved nav annotation source to use;
- whether placeholder route text is allowed only for loader smoke.

## Completion Conditions

This Root Task can move to `done` only when:

- candidate roots are reported separately;
- full clip list paths are written or inability is explained;
- component inventory artifacts are written;
- nav annotation source and gaps are documented;
- user-run dry-run download and monitoring scripts/commands are available;
- no training/download/checkpoint/source mutation has been executed.
