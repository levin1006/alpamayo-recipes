#!/usr/bin/env python3
"""Regenerate the compact Stage 1 nav-demo export from an existing checkpoint."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from alpamayo.data.pai_nav import PAIDatasetWithNav
from alpamayo.metrics import distance_metrics
from alpamayo.processor.qwen_processor import collate_fn_from_model_config
from alpamayo1_5_sft.models.sft_base_model import TrainableReasoningVLA


BASE_A1 = Path("/data/alpamayo_sft_artifacts/Alpamayo-1.5-10B-A1-format")
STAGE1_CKPT = Path(
    "/data/alpamayo_sft_artifacts/"
    "output_stage1_nav_smoke_stage1overfit300_20260623_104948/checkpoint-300"
)
DATASET_DIR = Path("/data/datasets/physical_ai_av")
SAMPLES_JSON = Path("/data/alpamayo_sft_artifacts/nav_demo_samples.json")
VLM_NAME = "Qwen/Qwen3-VL-8B-Instruct"
EXPECTED_NAV_DEMO_ROWS = 20

CHUNK_IDS = [
    214,
    224,
    276,
    317,
    420,
    727,
    728,
    968,
    982,
    1519,
    1657,
    1984,
    2277,
    2368,
    2372,
    2447,
    2599,
    2634,
    2868,
]

VLA_PREPROCESS_ARGS = {
    "_target_": "alpamayo.processor.qwen_processor.get_preprocess_data_fn_from_model_config",
    "chat_template_version": "r1_5",
    "components_order": ["image", "traj_history", "route", "prompt", "traj_future"],
    "components_prompt": ["traj_future"],
    "label_components": ["traj_future"],
    "include_camera_ids": True,
    "include_frame_nums": True,
    "generation_mode": True,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-p", type=float, default=0.98)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--num-traj-samples", type=int, default=1)
    parser.add_argument("--num-traj-sets", type=int, default=1)
    parser.add_argument("--max-generation-length", type=int, default=256)
    parser.add_argument("--dtype", choices=["bfloat16"], default="bfloat16")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: move_to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [move_to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(v, device) for v in value)
    return value


def load_rows() -> list[dict[str, Any]]:
    rows = json.loads(SAMPLES_JSON.read_text(encoding="utf-8"))
    if len(rows) != EXPECTED_NAV_DEMO_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_NAV_DEMO_ROWS} nav-demo rows, got {len(rows)}")
    for idx, row in enumerate(rows):
        row["row_index"] = idx
    return rows


def write_annotation(path: Path, row: dict[str, Any]) -> None:
    payload = [{k: v for k, v in row.items() if k != "row_index"}]
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_stage1(device: torch.device) -> TrainableReasoningVLA:
    model = TrainableReasoningVLA.from_alpamayo_checkpoint(
        checkpoint_path=str(STAGE1_CKPT),
        vlm_name_or_path=VLM_NAME,
    )
    model.to(device)
    model.eval()
    return model


def build_batch(
    model: TrainableReasoningVLA, annotation_path: Path, device: torch.device
) -> dict[str, Any]:
    dataset = PAIDatasetWithNav(
        annotations_path=str(annotation_path),
        local_dir=str(DATASET_DIR),
        chunk_ids=CHUNK_IDS,
        model_config=model.config,
        vla_preprocess_args=VLA_PREPROCESS_ARGS,
    )
    if len(dataset) != 1:
        raise RuntimeError(f"Expected one-row dataset for {annotation_path}, got {len(dataset)}")
    batch = collate_fn_from_model_config(
        [dataset[0]], model_config=model.config, chat_template_version="r1_5"
    )
    return move_to_device(batch, device)


def metric_values(pred_xyz: torch.Tensor, pred_rot: torch.Tensor, batch: dict[str, Any]) -> dict[str, float]:
    gt_xyz = batch["ego_future_xyz"][:, -1].detach().cpu()
    gt_rot = batch["ego_future_rot"][:, -1].detach().cpu()
    pred_xyz_cpu = pred_xyz.detach().cpu()
    pred_rot_cpu = pred_rot.detach().cpu()
    top_xyz = pred_xyz_cpu[:, :, :1]
    ade = distance_metrics.compute_ade(top_xyz, gt_xyz).squeeze(2).mean(-1)
    minade = distance_metrics.compute_minade(pred_xyz_cpu, gt_xyz, disable_summary=True)
    corner = distance_metrics.compute_grouped_corner_distance(
        pred_xyz_cpu,
        pred_rot_cpu,
        gt_xyz,
        gt_rot,
        torch.tensor((4.0, 3.0, 2.0), dtype=torch.float32),
        disable_summary=True,
    )
    return {
        "ade": float(ade[0].item()),
        "min_ade": float(minade["min_ade"][0].item()),
        "corner_distance": float(corner["corner_distance"][0].item()),
    }


def first_text(extra: dict[str, Any], key: str) -> str | None:
    if key not in extra:
        return None
    arr = np.asarray(extra[key], dtype=object).reshape(-1)
    return str(arr[0]) if arr.size else None


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        shutil.rmtree(args.output_root)
    model_dir = args.output_root / "stage1_sft"
    ann_dir = args.output_root / "_row_annotations"
    model_dir.mkdir(parents=True)
    ann_dir.mkdir(parents=True)

    rows = load_rows()
    device = torch.device(args.device)
    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    model = load_stage1(device)

    arrays: dict[str, np.ndarray] = {}
    records: list[dict[str, Any]] = []
    start = time.perf_counter()
    for row in rows:
        row_index = int(row["row_index"])
        annotation_path = ann_dir / f"row_{row_index:02d}.json"
        write_annotation(annotation_path, row)
        batch = build_batch(model, annotation_path, device)
        set_seed(args.seed)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            pred_xyz, pred_rot, extra = model.sample_trajectories_from_data(
                data=batch,
                top_p=args.top_p,
                temperature=args.temperature,
                num_traj_samples=args.num_traj_samples,
                num_traj_sets=args.num_traj_sets,
                max_generation_length=args.max_generation_length,
                return_extra=True,
            )
        prefix = f"row_{row_index:02d}"
        arrays[f"{prefix}/pred_xyz"] = pred_xyz.detach().cpu().numpy()
        arrays[f"{prefix}/pred_rot"] = pred_rot.detach().cpu().numpy()
        arrays[f"{prefix}/ego_future_xyz"] = batch["ego_future_xyz"].detach().cpu().numpy()
        arrays[f"{prefix}/ego_future_rot"] = batch["ego_future_rot"].detach().cpu().numpy()
        arrays[f"{prefix}/ego_history_xyz"] = batch["ego_history_xyz"].detach().cpu().numpy()
        arrays[f"{prefix}/ego_history_rot"] = batch["ego_history_rot"].detach().cpu().numpy()
        metrics = metric_values(pred_xyz, pred_rot, batch)
        endpoint_xy = arrays[f"{prefix}/pred_xyz"][0, 0, 0, -1, :2].astype(float).tolist()
        records.append(
            {
                "model": "stage1_sft",
                "row_index": row_index,
                "clip_id": row["clip_id"],
                "t0_relative": row.get("t0_relative"),
                "nav_text": row["nav_text"],
                "ade": metrics["ade"],
                "min_ade": metrics["min_ade"],
                "corner_distance": metrics["corner_distance"],
                "endpoint_xy": endpoint_xy,
                "npz_key_prefix": prefix,
                "cot": first_text(extra, "cot") if isinstance(extra, dict) else None,
                "pred_answer": first_text(extra, "pred_answer") if isinstance(extra, dict) else None,
                "seed": args.seed,
                "top_p": args.top_p,
                "temperature": args.temperature,
                "num_traj_samples": args.num_traj_samples,
                "num_traj_sets": args.num_traj_sets,
                "max_generation_length": args.max_generation_length,
                "dtype": args.dtype,
                "load_contract": "Stage1 VLM-side TrainableReasoningVLA checkpoint via recipes contract",
            }
        )

    np.savez_compressed(model_dir / "predictions.npz", **arrays)
    with (model_dir / "results.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "model": "stage1_sft",
        "num_rows": len(records),
        "mean_ade": float(np.mean([r["ade"] for r in records])),
        "median_ade": float(np.median([r["ade"] for r in records])),
        "mean_min_ade": float(np.mean([r["min_ade"] for r in records])),
        "median_min_ade": float(np.median([r["min_ade"] for r in records])),
        "mean_corner_distance": float(np.mean([r["corner_distance"] for r in records])),
        "median_corner_distance": float(np.median([r["corner_distance"] for r in records])),
        "runtime_s": round(time.perf_counter() - start, 3),
        "peak_vram_mib": (
            int(torch.cuda.max_memory_allocated(device) / 1024 / 1024)
            if torch.cuda.is_available()
            else None
        ),
    }
    (model_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    snapshot = [{k: v for k, v in row.items() if k != "row_index"} for row in rows]
    (args.output_root / "annotations_snapshot.json").write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    manifest = {
        "created_for": "canonical_stage1_export_regeneration",
        "note": "Baseline directory intentionally omitted. Previous recipes baseline was mismatched; canonical baseline should use the 2026-06-18 matched baseline artifacts.",
        "paths": {
            "baseline_a1_reference": str(BASE_A1),
            "stage1_checkpoint": str(STAGE1_CKPT),
            "samples_json": str(SAMPLES_JSON),
        },
        "settings": {
            "seed": args.seed,
            "top_p": args.top_p,
            "temperature": args.temperature,
            "num_traj_samples": args.num_traj_samples,
            "num_traj_sets": args.num_traj_sets,
            "max_generation_length": args.max_generation_length,
            "dtype": args.dtype,
        },
        "load_contracts": {
            "stage1_sft": "TrainableReasoningVLA.from_alpamayo_checkpoint(STAGE1_CKPT, vlm_name_or_path=Qwen/Qwen3-VL-8B-Instruct)",
            "baseline": "not included; use canonical 2026-06-18 matched baseline for visualization",
        },
        "summary": summary,
        "evaluation_policy": {
            "default_unit": "full_20_rows",
            "row07_note": "row07 is one historical reproducibility reference among 20 rows, not a gate or primary decision row.",
        },
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"summary": summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
