---
doc_type: report
status: active
created_at: 2026-06-26 07:47:54
updated_at: 2026-06-26 08:20:00
---

# Full Joint SFT Stop Decision

## Decision

`stage1-sft + stage2-sft joint full update` 실험은 여기서 중단한다.

이 실험은 현재 확인한 Alpamayo-R1 저자 canonical path가 아니며, 5개 H100
GPU를 모두 사용해도 현재 코드/설정으로는 안정적으로 학습에 진입하지
못했다. 따라서 더 진행하면 환경 검증보다 비canonical 학습 전략 디버깅으로
범위가 바뀐다.

## Current Source State

full joint gate를 위해 임시로 적용했던 autocast/ZeRO 실험성 변경은 복구했다.
다만 `stage1-sft-frozen + stage2 sft` 실험을 재현하려면 Stage1 VLM checkpoint가
Stage2 모델의 nested VLM에 실제로 로드되어야 하므로, 그 loader contract는
source에 보존한다.

보존하는 최소 source contract:

- `recipes/alpamayo1_5_sft/models/sft_alpamayo_r1.py`
  - `from_pretrained(..., stage1_vlm_checkpoint_path=...)`에서 base checkpoint
    로드 후 Stage1 VLM checkpoint를 다시 로드
  - `cotrain_vlm=false`일 때 VLM freeze 유지
- `recipes/alpamayo1_5_sft/models/sft_base_model.py`
  - Stage1 checkpoint의 `vlm.*` key를 nested VLM에 맞게 prefix strip
  - loader log에 `stripped_vlm_prefix`, `missing`, `unexpected` 출력

보존하지 않는 full joint-only source changes:

- action/expert autocast 실험 patch
- ZeRO-3 load compatibility patch
- gradient checkpointing wrapper patch

따라서 현재 source는 “원본으로 완전 회귀”가 아니라, 의미 있는 Stage2 재현에
필요한 loader fix만 남긴 상태다.

## Full Joint Runtime Evidence

5 GPU full joint 실행 시도:

- run id: `stage1_sft_stage2_sft_joint_20260626_064857`
- log:
  `/workspace/alpamayo-recipes/logs/sft_runs/stage1_sft_stage2_sft_joint_20260626_064857.log`
- output root:
  `/data/alpamayo_sft_artifacts/output_stage1_sft_stage2_sft_joint_20260626_064857`
- GPUs: `0,1,2,3,4`
- trainable params: `11,078,526,194`
- Stage1 VLM load evidence:
  `Loaded 750 VLM tensors ... stripped_vlm_prefix=750, missing=0, unexpected=0`

Failure:

- step 1 진입 후 backward 중 rank4에서 CUDA OOM
- 추가 할당 실패: `1.87 GiB`
- GPU4 상태: total `93.10 GiB`, free `895.38 MiB`, process memory
  `92.20 GiB`
- checkpoint 미생성

ZeRO-3 임시 gate:

- run id: `stage1_sft_stage2_sft_joint_zero3_gate2_20260626_065222`
- log:
  `/workspace/alpamayo-recipes/logs/sft_runs/stage1_sft_stage2_sft_joint_zero3_gate2_20260626_065222.log`
- result: 실패
- reason: ZeRO-3 partitioned tensor 상태에서 Stage1 VLM checkpoint 로드가
  `torch.Size([0])` shape mismatch와 충돌

Interpretation:

- ZeRO-2: 5 GPU에서도 full trainable backward memory가 부족
- ZeRO-3: 현재 Stage1 VLM load path와 호환되지 않음
- gradient checkpointing: 현재 `TrainableAlpamayoR1` wrapper가 Trainer의
  gradient checkpointing hook을 지원하지 않아 바로 켤 수 없음

## Canonical Path Assessment

Alpamayo-R1 저자 흐름은 full joint update가 아니라 staged contract에 가깝다.
특히 Stage2/action expert 쪽은 VLM에서 나온 cache를 사용하되, action expert
loss가 VLM 전체로 그대로 역전파되는 full joint update를 canonical 방법으로
제시하지 않는다.

따라서 현재 full joint 실험은 다음처럼 해석한다.

- 의미 있음: 시스템 한계와 full joint 비현실성을 확인한 diagnostic gate
- 의미 없음: 저자 recipe 재현성 또는 canonical 성능 비교의 후속 실험
- 다음 우선순위: full joint 재시도가 아니라 selected VLM layer, LoRA/adapters,
  또는 action expert-only objective/rollout contract 개선 설계

## Visualization Data Readiness

PM/nav visualization thread가 기존 renderer로 사용할 수 있는 compact export와
overlay 결과는 이미 존재한다.

### Export Script Inventory

Reusable export scripts are repository-managed under `scripts/`, not under
`logs/`.

Current tracked-script candidates:

- `scripts/nav_demo_exports/export_stage1_compact.py`
- `scripts/nav_demo_exports/export_stage2_compact.py`
- `scripts/nav_demo_exports/export_stage1_vlm_baseline_expert.py`
- `scripts/sft_experiments/run_stage1_sft_frozen_stage2_sft.sh`
- `scripts/sft_experiments/archive/run_stage1_sft_frozen_stage2_sft_20260625_190024.sh`

The `nav_demo_exports` scripts correspond to the compact export contracts used
for the current Stage1/Stage2 comparison artifacts.

The `run_stage1_sft_frozen_stage2_sft.sh` script preserves the meaningful
`stage1-sft-frozen + stage2 sft` training contract as a reusable reproduction
entrypoint. The archive script preserves the historical launcher shape for the
2026-06-25 run as evidence, not as the preferred way to launch new reruns.

The following experiment-local helper was not promoted because it was a
historical one-row helper and is superseded by the compact export and overlay
artifacts:

- `logs/stage1_row07_gate/run_row07_gate.py`

`logs/` remains an ignored runtime evidence area for stdout/stderr logs,
`.done/.failed` markers, cache files, and generated snapshots. Documents should
not depend on ignored `logs/**/*.py` or `logs/**/*.sh` paths as canonical
reproduction steps.

### Compact Exports

| Label | Root | Required files | Status |
| --- | --- | --- | --- |
| `stage1-sft-only` | `/home/user/Workspace/alpamayo1.5/experiments/nav_demo_inference_comparison/2026-06-23/stage1_sft_vs_baseline/stage1_export` | `manifest.json`, `annotations_snapshot.json`, `stage1_sft/results.jsonl`, `stage1_sft/predictions.npz`, `stage1_sft/summary.json` | ready |
| `stage1-baseline-frozen + stage2 sft` | `/home/user/Workspace/alpamayo1.5/experiments/nav_demo_inference_comparison/2026-06-24/stage2_sft_vs_baseline/stage2_export` | `manifest.json`, `annotations_snapshot.json`, `baseline/*`, `stage2_sft/*` | ready |
| `stage1-sft-frozen + stage2 sft` | `/home/user/Workspace/alpamayo1.5/experiments/nav_demo_inference_comparison/2026-06-25/stage1_sft_frozen_stage2_sft_vs_baseline/stage2_export` | `manifest.json`, `annotations_snapshot.json`, `baseline/*`, `stage1_sft_frozen_stage2_sft/*` | ready |

Each model directory above contains:

- `results.jsonl`
- `predictions.npz`
- `summary.json`

The `_row_annotations` directories are annotation snapshots, not model result
directories, so they are not expected to contain `results.jsonl` or
`predictions.npz`.

### Overlay Outputs

| Label | Root | PNG count | Status |
| --- | --- | ---: | --- |
| `stage1-sft-only vs matched baseline` | `/home/user/Workspace/alpamayo1.5/experiments/nav_demo_inference_comparison/2026-06-24/stage1_sft_vs_matched_baseline/comparison_visuals` | `20` | ready |
| `stage1-baseline-frozen + stage2 sft vs matched baseline` | `/home/user/Workspace/alpamayo1.5/experiments/nav_demo_inference_comparison/2026-06-24/stage2_sft_vs_matched_baseline/comparison_visuals` | `20` | ready |
| `stage1-sft-frozen + stage2 sft vs matched baseline` | `/home/user/Workspace/alpamayo1.5/experiments/nav_demo_inference_comparison/2026-06-25/stage1_sft_frozen_stage2_sft_vs_matched_baseline/comparison_visuals` | `20` | ready |

Each overlay root contains:

- `summary.json`
- `metrics.csv`
- 20 row-level PNG visualizations

## Current Comparison Summary

Current four-combination interpretation:

1. matched official baseline: mean ADE `1.751832239329815`
2. `stage1-sft-only`: mean ADE `0.09993635825812816`
3. `stage1-baseline-frozen + stage2 sft`: mean ADE `1.4155602216720582`
4. `stage1-sft-frozen + stage2 sft`: mean ADE `1.4630816221237182`

Important caveat:

- `stage1-sft-only` is a VLM discrete trajectory token reference and should not
  be treated as the final continuous action expert output.
- The two Stage2 variants are the meaningful continuous action expert outputs
  available from the current completed experiments.
- full joint has no valid checkpoint or compact export because training failed
  before checkpoint creation.

## Next Decision

Do not resume full joint training without a new plan. If joint adaptation remains
necessary, the next plan should be partial joint rather than full update:

- selected top-k VLM layers + action expert
- VLM LoRA/adapters + action expert
- action expert-only with improved rollout/objective contract

Before any new runtime:

- define trainable parameter scope;
- estimate memory;
- confirm loader/freeze/equality gates;
- complete the full 20-row export before reporting demo evaluation complete.
