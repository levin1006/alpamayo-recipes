#!/usr/bin/env python3
"""Export compact nav-demo predictions for the Stage 2 review gate."""

from __future__ import annotations

import argparse
import json
import math
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
from alpamayo1_5_sft.models.sft_alpamayo_r1 import TrainableAlpamayoR1


BASE_A1 = Path("/data/alpamayo_sft_artifacts/Alpamayo-1.5-10B-A1-format")
RAW_BASE = Path("/data/alpamayo_sft_artifacts/Alpamayo-1.5-10B")
STAGE1_CKPT = Path(
    "/data/alpamayo_sft_artifacts/"
    "output_stage1_nav_smoke_stage1overfit300_20260623_104948/checkpoint-300"
)
STAGE2_CKPT = Path(
    "/data/alpamayo_sft_artifacts/"
    "output_stage2_nav_overfit300_stage2overfit300_plan_20260623_144245/checkpoint-300"
)
DATASET_DIR = Path("/data/datasets/physical_ai_av")
SAMPLES_JSON = Path("/data/alpamayo_sft_artifacts/nav_demo_samples.json")

OLD_ROW07_ENDPOINT_XY = np.asarray([33.33718490600586, -3.5824496746063232], dtype=np.float64)
OLD_ROW07_ADE = 0.5370676517486572
ROW07_INDEX = 7
ROW07_NAV_TEXT = "Turn right in 30m"

CHUNK_IDS = [214, 224, 276, 317, 420, 727, 728, 968, 982, 1519, 1657, 1984, 2277, 2368, 2372, 2447, 2599, 2634, 2868]

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
    parser.add_argument("--stage2-ckpt", type=Path, default=STAGE2_CKPT)
    parser.add_argument("--stage2-name", default="stage2_sft")
    parser.add_argument("--created-for", default="stage2_sft_vs_baseline_compact_export")
    parser.add_argument("--rows", default="7", help="'7' for gate or 'all' for 20-row export.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-p", type=float, default=0.98)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--diffusion-temperature", type=float, default=0.6)
    parser.add_argument("--num-traj-samples", type=int, default=1)
    parser.add_argument("--num-traj-sets", type=int, default=1)
    parser.add_argument("--max-generation-length", type=int, default=256)
    parser.add_argument("--dtype", choices=["bfloat16"], default="bfloat16")
    return parser.parse_args()


def ensure_safe_output_root(output_root: Path) -> None:
    resolved = output_root.resolve()
    protected = {
        Path("/").resolve(),
        Path("/data").resolve(),
        Path("/data/alpamayo_sft_artifacts").resolve(),
        BASE_A1.resolve(),
        RAW_BASE.resolve(),
        STAGE1_CKPT.resolve(),
        STAGE1_CKPT.parent.resolve(),
        DATASET_DIR.resolve(),
    }
    if resolved in protected:
        raise RuntimeError(f"Refusing unsafe output root: {resolved}")


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
    for idx, row in enumerate(rows):
        row["row_index"] = idx
    if rows[ROW07_INDEX]["nav_text"] != ROW07_NAV_TEXT:
        raise RuntimeError(f"Unexpected row07 nav_text: {rows[ROW07_INDEX]['nav_text']!r}")
    return rows


def selected_rows(rows: list[dict[str, Any]], selector: str) -> list[dict[str, Any]]:
    if selector == "all":
        return rows
    indexes = [int(part) for part in selector.split(",") if part]
    return [rows[index] for index in indexes]


def write_annotation(path: Path, row: dict[str, Any]) -> None:
    payload = [{k: v for k, v in row.items() if k != "row_index"}]
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_model(path: Path, device: torch.device) -> TrainableAlpamayoR1:
    model = TrainableAlpamayoR1.from_pretrained(str(path))
    model.to(device)
    model.eval()
    return model


def build_batch(
    model: TrainableAlpamayoR1, annotation_path: Path, device: torch.device
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
    minade_dict = distance_metrics.compute_minade(pred_xyz_cpu, gt_xyz, disable_summary=True)
    corner_dict = distance_metrics.compute_grouped_corner_distance(
        pred_xyz_cpu,
        pred_rot_cpu,
        gt_xyz,
        gt_rot,
        torch.tensor((4.0, 3.0, 2.0), dtype=torch.float32),
        disable_summary=True,
    )
    result = {
        "ade": float(ade[0].item()),
        "min_ade": float(minade_dict["min_ade"][0].item()),
    }
    if "corner_distance" in corner_dict:
        result["corner_distance"] = float(corner_dict["corner_distance"][0].item())
    return result


def first_text(extra: dict[str, Any], key: str) -> str | None:
    if key not in extra:
        return None
    arr = np.asarray(extra[key], dtype=object).reshape(-1)
    return str(arr[0]) if arr.size else None


def run_model(
    model_name: str,
    model_path: Path,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    output_root: Path,
) -> dict[str, Any]:
    device = torch.device(args.device)
    model_dir = output_root / model_name
    ann_dir = output_root / "_row_annotations"
    model_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    model = load_model(model_path, device)

    records: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
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
                diffusion_kwargs={"temperature": args.diffusion_temperature},
                return_extra=True,
            )

        metrics = metric_values(pred_xyz, pred_rot, batch)
        prefix = f"row_{row_index:02d}"
        arrays[f"{prefix}/pred_xyz"] = pred_xyz.detach().cpu().numpy()
        arrays[f"{prefix}/pred_rot"] = pred_rot.detach().cpu().numpy()
        arrays[f"{prefix}/ego_future_xyz"] = batch["ego_future_xyz"].detach().cpu().numpy()
        arrays[f"{prefix}/ego_future_rot"] = batch["ego_future_rot"].detach().cpu().numpy()
        arrays[f"{prefix}/ego_history_xyz"] = batch["ego_history_xyz"].detach().cpu().numpy()
        arrays[f"{prefix}/ego_history_rot"] = batch["ego_history_rot"].detach().cpu().numpy()

        pred_np = arrays[f"{prefix}/pred_xyz"]
        endpoint_xy = pred_np[0, 0, 0, -1, :2].astype(float).tolist()
        record = {
            "model": model_name,
            "row_index": row_index,
            "clip_id": row["clip_id"],
            "t0_relative": row.get("t0_relative"),
            "nav_text": row["nav_text"],
            "ade": metrics["ade"],
            "min_ade": metrics["min_ade"],
            "corner_distance": metrics.get("corner_distance"),
            "endpoint_xy": endpoint_xy,
            "npz_key_prefix": prefix,
            "cot": first_text(extra, "cot") if isinstance(extra, dict) else None,
            "pred_answer": first_text(extra, "pred_answer") if isinstance(extra, dict) else None,
            "seed": args.seed,
            "top_p": args.top_p,
            "temperature": args.temperature,
            "diffusion_temperature": args.diffusion_temperature,
            "num_traj_samples": args.num_traj_samples,
            "num_traj_sets": args.num_traj_sets,
            "max_generation_length": args.max_generation_length,
            "dtype": args.dtype,
            "load_contract": (
                "baseline A1-format TrainableAlpamayoR1"
                if model_name == "baseline"
                else "Stage2 TrainableAlpamayoR1 checkpoint via recipes sft_stage2_nav contract"
            ),
        }
        records.append(record)

    np.savez_compressed(model_dir / "predictions.npz", **arrays)
    with (model_dir / "results.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = summarize(model_name, records, runtime_s=time.perf_counter() - start, device=device)
    (model_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def summarize(model_name: str, records: list[dict[str, Any]], runtime_s: float, device: torch.device) -> dict[str, Any]:
    ades = [r["ade"] for r in records if r.get("ade") is not None and math.isfinite(r["ade"])]
    min_ades = [r["min_ade"] for r in records if r.get("min_ade") is not None and math.isfinite(r["min_ade"])]
    corners = [
        r["corner_distance"]
        for r in records
        if r.get("corner_distance") is not None and math.isfinite(r["corner_distance"])
    ]
    return {
        "model": model_name,
        "num_rows": len(records),
        "mean_ade": float(np.mean(ades)) if ades else None,
        "mean_min_ade": float(np.mean(min_ades)) if min_ades else None,
        "mean_corner_distance": float(np.mean(corners)) if corners else None,
        "runtime_s": round(runtime_s, 3),
        "peak_vram_mib": (
            int(torch.cuda.max_memory_allocated(device) / 1024 / 1024)
            if torch.cuda.is_available()
            else None
        ),
    }


def write_manifest(output_root: Path, args: argparse.Namespace, rows: list[dict[str, Any]], summaries: dict[str, Any]) -> None:
    manifest = {
        "created_for": args.created_for,
        "rows": [int(row["row_index"]) for row in rows],
        "settings": {
            "seed": args.seed,
            "top_p": args.top_p,
            "temperature": args.temperature,
            "diffusion_temperature": args.diffusion_temperature,
            "num_traj_samples": args.num_traj_samples,
            "num_traj_sets": args.num_traj_sets,
            "max_generation_length": args.max_generation_length,
            "dtype": args.dtype,
        },
        "paths": {
            "raw_baseline": str(RAW_BASE),
            "baseline_a1": str(BASE_A1),
            "stage1_checkpoint_reference": str(STAGE1_CKPT),
            "stage2_checkpoint": str(args.stage2_ckpt),
            "samples_json": str(SAMPLES_JSON),
        },
        "load_contracts": {
            "baseline": "TrainableAlpamayoR1.from_pretrained(BASE_A1)",
            "stage1_sft": "VLM-side TrainableReasoningVLA checkpoint; not exported here as full official Alpamayo1_5",
            args.stage2_name: "TrainableAlpamayoR1.from_pretrained(args.stage2_ckpt)",
        },
        "summaries": summaries,
        "row07_official_reference": {
            "endpoint_xy": OLD_ROW07_ENDPOINT_XY.tolist(),
            "ade": OLD_ROW07_ADE,
        },
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    snapshot = [{k: v for k, v in row.items() if k != "row_index"} for row in rows]
    (output_root / "annotations_snapshot.json").write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def row07_gate(output_root: Path) -> dict[str, Any]:
    baseline_path = output_root / "baseline" / "results.jsonl"
    stage2_candidates = [p for p in output_root.iterdir() if p.is_dir() and p.name != "baseline" and not p.name.startswith("_")]
    if len(stage2_candidates) != 1:
        raise RuntimeError(f"Expected exactly one non-baseline model directory, got {stage2_candidates}")
    stage2_path = stage2_candidates[0] / "results.jsonl"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8").splitlines()[0])
    stage2 = json.loads(stage2_path.read_text(encoding="utf-8").splitlines()[0])
    baseline_endpoint = np.asarray(baseline["endpoint_xy"], dtype=np.float64)
    stage2_endpoint = np.asarray(stage2["endpoint_xy"], dtype=np.float64)
    return {
        "old_official_endpoint_xy": OLD_ROW07_ENDPOINT_XY.tolist(),
        "old_official_ade": OLD_ROW07_ADE,
        "baseline_endpoint_xy": baseline["endpoint_xy"],
        "baseline_ade": baseline["ade"],
        "baseline_endpoint_l2_vs_old": float(np.linalg.norm(baseline_endpoint - OLD_ROW07_ENDPOINT_XY)),
        "baseline_ade_abs_diff_vs_old": abs(float(baseline["ade"]) - OLD_ROW07_ADE),
        "stage2_endpoint_xy": stage2["endpoint_xy"],
        "stage2_ade": stage2["ade"],
        "stage2_endpoint_l2_vs_old": float(np.linalg.norm(stage2_endpoint - OLD_ROW07_ENDPOINT_XY)),
        "stage2_ade_abs_diff_vs_old": abs(float(stage2["ade"]) - OLD_ROW07_ADE),
        "baseline_nav_text": baseline["nav_text"],
        "stage2_nav_text": stage2["nav_text"],
    }


def main() -> None:
    args = parse_args()
    rows = selected_rows(load_rows(), args.rows)
    ensure_safe_output_root(args.output_root)
    if args.output_root.exists():
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True)
    summaries = {
        "baseline": run_model("baseline", BASE_A1, rows, args, args.output_root),
        args.stage2_name: run_model(args.stage2_name, args.stage2_ckpt, rows, args, args.output_root),
    }
    write_manifest(args.output_root, args, rows, summaries)
    gate = row07_gate(args.output_root) if [r["row_index"] for r in rows] == [ROW07_INDEX] else None
    if gate is not None:
        (args.output_root / "row07_gate.json").write_text(
            json.dumps(gate, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(gate, indent=2, ensure_ascii=False))
    else:
        print(json.dumps({"summaries": summaries}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
