---
doc_type: task
task_role: root
status: superseded
task_id: track-c-pai-sft-demo-root
execution_session_id: 019eed1d-b6bc-7562-880e-f198c3b5f67b
managing_session_id: 019ecfca-468c-7682-b75e-aadca21dfe86
parent_plan: docs/2026-06-22 [Plan] Track C PAI SFT Demo Gate Before a2z.md
parent_task: docs/2026-06-17 [Task] Track C a2z SFT Readiness Root Task.md
created_at: 2026-06-22 11:17:55 KST
updated_at: 2026-06-22 12:56:51 KST
---

# Track C PAI SFT Demo Root Task

## Purpose

Track the execution of the PAI nav demo gate before returning to Track C a2z
SFT readiness. This task is owned by the execution session. The separate Track C
총괄 session remains responsible for plan approval, result review, and a2z
return decisions.

## Parent Plan

- `docs/2026-06-22 [Plan] Track C PAI SFT Demo Gate Before a2z.md`

## Current Roll-Up

Status: `superseded`

P0 preflight and P1 no-training batch proof passed. No training, checkpoint
creation, or model loading was performed. This task is superseded by the
bounded Stage 1 overfit smoke task after the user selected that as the next
gate.

## Coordination

- Execution session: `019eed1d-b6bc-7562-880e-f198c3b5f67b`
- Managing session: `019ecfca-468c-7682-b75e-aadca21dfe86`
- Completion results must be relayed to the managing session before final
  closure.

## Retired Runnable Entrypoints

The original P0/P1 wrapper scripts were retired during the Stage 1 overfit
smoke cleanup. P0/P1 evidence remains preserved in
`docs/2026-06-22 [Report] Track C PAI SFT Demo Result.md`; reruns should use
the current `scripts/a15_sft_readme_*` sequence or a new task-specific runner.

## Required Subtask Registry

| Subtask | Status | First action |
| --- | --- | --- |
| P0 preflight | done | PAI root, nav JSON, GPU/env, DeepSpeed, W&B policy checked; A1-format checkpoint still unresolved |
| P1 no-training batch proof | done | Dataset train/val sample load and collate proof passed |
| P2 Stage 1 bounded overfit smoke | blocked | Start only after separate user approval and verified A1-format checkpoint |
| P3 result report | done | Evidence, inference, unknowns, commands, logs, artifacts, and non-claims documented |
| P4 optional Stage 2/eval | not_started | Start only after Stage 1 review and separate user approval |

## Blockers

- Stage 1 requires a later explicit approval even though P0/P1 passed.
- Converted Alpamayo 1.5 A1-format checkpoint path is not yet verified. A raw
  cached `nvidia/Alpamayo-1.5-10B` snapshot exists, but its config is
  `model_type: alpamayo1_5`, not the README-required A1-format conversion.

## Execution Log

- 2026-06-22 11:17:55 KST: Draft Plan and Root Task created for user review.
  No runtime command, training command, checkpoint creation, or model loading
  performed.
- 2026-06-22 11:19:45 KST: Self-review completed. The draft keeps Plan
  approval limited to P0 preflight and P1 no-training batch proof, keeps Stage 1
  under separate approval, keeps Stage 2/eval under a later approval, and keeps
  non-claims explicit.
- 2026-06-22 11:20:11 KST: User approved the Plan and instructed a separate
  execution session. Administrative state updated to `active`/`canonical: true`.
  Approval remains limited to P0 preflight and P1 no-training batch proof.
- 2026-06-22 11:23:07 KST: User provided execution session ID and managing
  session ID. Completion result routing recorded; no completion-result message
  sent yet because P0/P1 evidence is still pending.
- 2026-06-22 11:27:23 KST: Added no-argument virtual environment setup and
  P0/P1 runner entrypoints. Syntax check passed with `bash -n`; the scripts were
  not executed because `uv sync` and data proof should run in the execution
  session.
- 2026-06-22 11:28:00 KST: P0 preflight and P1 no-training batch proof passed.
  Result report created at
  `docs/2026-06-22 [Report] Track C PAI SFT Demo Result.md`. No training,
  checkpoint creation, or model loading performed.
- 2026-06-22 11:30:18 KST: Completion result relayed to managing session
  `019ecfca-468c-7682-b75e-aadca21dfe86`.
- 2026-06-22 12:56:51 KST: Superseded by the bounded Stage 1 20-sample overfit
  smoke task. P0/P1 evidence remains valid historical input; this task no
  longer owns the next execution gate.

## Next Decision

Review P0/P1 result report, then decide whether to approve Stage 1 bounded
overfit smoke after supplying or approving an A1-format checkpoint path.

## Managing Session Review

2026-06-22 11:30:39 KST review result: P0/P1 evidence accepted by the managing
session.

Accepted evidence:

- `/mnt/zfs_pool/physical_ai_av` has the required metadata and 19/19 checked nav
  chunks complete for the required components.
- `nav_demo_samples.json` remains a 20-row / 19-unique-clip demo annotation set.
- `PAIDatasetWithNav` train and val lengths are both 20.
- Train and val tokenized text include route start token and `nav_text`.
- Train `labels_mask` matches the `traj_future` span plus assistant EOS, with
  zero route-span masked tokens.
- Val `labels_mask` is all false.
- No training, checkpoint creation, model instantiation, or model weight loading
  was performed.

Review decision:

- P0/P1 are accepted as a valid official-recipe wiring proof.
- P2 remains blocked until the user explicitly approves Stage 1 and provides or
  approves a verified A1-format checkpoint path.
- This proof must not be promoted to broad PAI nav readiness, a2z SFT readiness,
  model-quality evidence, or generalization evidence.

## Completion Conditions

This Root Task can move to `done` only when:

- P0 preflight is recorded;
- P1 no-training batch proof is recorded;
- any Stage 1 run, if approved, is capped and reported;
- Stage 2/eval is either explicitly completed or explicitly deferred;
- the final report separates Evidence, Inference, and Unknowns;
- no broad PAI nav readiness or a2z readiness claim is made from this demo.
