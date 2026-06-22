---
doc_type: handoff
status: ready
created_at: 2026-06-22 12:56:51 KST
source_session_id: 019ecfca-468c-7682-b75e-aadca21dfe86
target_role: execution_thread
parent_plan: docs/2026-06-22 [Plan] Track C PAI 20 Sample Stage 1 Overfit Smoke.md
parent_task: docs/2026-06-22 [Task] Track C PAI 20 Sample Stage 1 Overfit Smoke Root Task.md
---

# Track C PAI 20 Sample Stage 1 Overfit Smoke Execution Handoff

## Prompt For New Thread

```text
현재 작업 디렉터리는 /home/user/Workspace/alpamayo-recipes 이다.

역할:
- 너는 Track C PAI 20-sample Stage 1 overfit smoke 실행 담당 세션이다.
- 사용자가 직접 대화할 수 있는 실행 스레드로 동작한다.
- 결과 보고는 Track C manager session 019ecfca-468c-7682-b75e-aadca21dfe86 로 하라.
- PM routing이 필요하면 현재 PM target 019eed40-e996-75b3-aa65-916422226066 도 함께 명시하라.

참조 문서:
- docs/2026-06-22 [Plan] Track C PAI 20 Sample Stage 1 Overfit Smoke.md
- docs/2026-06-22 [Task] Track C PAI 20 Sample Stage 1 Overfit Smoke Root Task.md
- docs/2026-06-22 [Report] Track C PAI SFT Demo Result.md

목표:
- PAI nav demo 20 rows / 19 clips에 대해 Stage 1 SFT가 bounded overfit smoke로 동작하는지 확인한다.
- 이 실험은 성능/일반화 증명이 아니라, 작은 데이터도 학습하지 못하면 recipe 배선이 잘못됐다는 sanity check이다.
- 기대 결과는 finite loss, loss 감소, trainer checkpoint artifact 생성이다.

중요 해석:
- 같은 20개로 학습하고 같은 20개에서 ADE가 좋아지는 것은 당연할 수 있으므로 성능 증거가 아니다.
- 결과는 오직 recipe wiring / overfit smoke evidence로 분류하라.
- a2z readiness, broad PAI readiness, model quality improvement, autonomous-driving safety improvement를 주장하지 마라.

사용할 스크립트:
- scripts/a15_sft_readme_00_setup_env.sh
- scripts/a15_sft_readme_01_download_pai_nav_chunks.sh
- scripts/a15_sft_readme_02_download_checkpoint.sh
- scripts/a15_sft_readme_03_convert_checkpoint_to_a1.sh
- scripts/a15_sft_readme_04_verify_a1_checkpoint.sh
- scripts/a15_sft_readme_05_stage1_nav_smoke.sh

기본 경로:
- PAI root: /mnt/zfs_pool/physical_ai_av
- nav annotations: /home/user/Workspace/alpamayo1.5/notebooks/nav_demo_samples.json
- artifact root: /mnt/zfs_pool/alpamayo_sft_artifacts

실행 정책:
- 모든 step script는 기본적으로 tmux에서 실행된다.
- training을 시작하는 05 script는 RUN_STAGE1 입력 확인이 필요하다.
- 기본 Stage 1은 max_steps=20, WANDB_MODE=disabled, trainer.report_to=none, CUDA_VISIBLE_DEVICES=0, NPROC_PER_NODE=1 이다.
- Stage 2/eval은 이번 지시 범위가 아니다. 사용자와 Track C manager review 없이 실행하지 마라.
- git commit/push 하지 마라. 커밋은 manager 세션이 수행한다.

필수 보고 형식:
보고 작성 시각(KST):
보고 대상 Track C manager session: 019ecfca-468c-7682-b75e-aadca21dfe86
실행한 스크립트:
환경 override:
A1-format checkpoint path:
Stage 1 output dir:
loss/log/checkpoint evidence:
Evidence:
Inference:
Unknowns:
Non-claims:
필요한 사용자 결정:
상태: DONE / BLOCKED / FAILED / NEEDS_USER_DECISION
```

## Notes

이 handoff 문서는 현재 앱 세션에서 새 Codex thread 생성 도구가 노출되지
않는 경우를 대비한 전달용 기준 문서다.
