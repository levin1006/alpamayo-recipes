---
doc_type: task
task_role: root
status: needs_review
task_id: track-c-a2z-sft-readiness-root
parent_plan: docs/2026-06-17 [Plan] Track C a2z SFT Feasibility and Implementation Readiness.md
parent_task:
created_at: 2026-06-17 14:49:50 KST
updated_at: 2026-06-22 12:56:51 KST
---

# Track C a2z SFT Readiness Root Task

## Purpose

Track execution against the approved Track C a2z SFT readiness Plan. This task
does not authorize training. It tracks feasibility explanation, user review,
implementation preparation, no-training materialization checks, and the later
decision gate for any micro SFT smoke.

## Canonical Plan

- `docs/2026-06-17 [Plan] Track C a2z SFT Feasibility and Implementation Readiness.md`

## Current Roll-Up

Status: `needs_review`

Subtask F0 has produced the learning/review artifact and still needs user
review. The separate PAI demo gate passed P0/P1 no-training proof and has been
superseded by a bounded 20-sample Stage 1 overfit-smoke gate under
`docs/2026-06-22 [Plan] Track C PAI 20 Sample Stage 1 Overfit Smoke.md`. No
a2z implementation code or a2z training has started in the parent Track C path.

## Required Subtask Registry

| Subtask | Status | First action |
| --- | --- | --- |
| F0 learning/review artifact | needs_review | User reviews learning guide and answers review gate |
| F1 a2z source inventory | not_started | Define tiny source manifest and row identity |
| F2 split/leakage policy | not_started | Encode log-level split rules and failure signals |
| F3 no-training materialization check | not_started | Plan dataset/processor/collator compatibility proof |
| F4 micro SFT smoke proposal | not_started | Draft later runtime envelope without authorization |
| PAI 20-sample overfit gate | not_started | Verify A1-format checkpoint, then run bounded Stage 1 smoke in separate execution thread |
| I0-I5 implementation readiness | not_started | Start only after feasibility review acceptance |

## Blockers

- User understanding/review gate not yet completed.
- No implementation task has been authorized beyond the approved planning
  baseline.
- a2z training, a2z implementation, and source dataset mutation remain
  forbidden.
- PAI Stage 1 overfit smoke is approved only as a bounded demo payload wiring
  check and still requires verified A1-format checkpoint before launch.

## Next Decision

Review the F0 learning guide and decide whether it is sufficient for the user
understanding gate:

`docs/2026-06-17 [Note] Track C a2z SFT Training Structure Learning Guide.md`

Then review or delegate the PAI 20-sample overfit smoke gate:

`docs/2026-06-22 [Plan] Track C PAI 20 Sample Stage 1 Overfit Smoke.md`

## Completion Conditions

This Root Task can move to `done` only when:

- the user-facing SFT learning/review artifact is complete and reviewed;
- a2z source inventory and split policy are implementation-ready;
- no-training materialization check scope and failure signals are defined;
- implementation subtasks are either completed or explicitly deferred;
- any later micro SFT smoke remains separately approved before execution.
