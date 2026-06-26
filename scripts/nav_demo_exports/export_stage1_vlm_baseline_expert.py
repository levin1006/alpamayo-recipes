#!/usr/bin/env python3
"""Export 20-row predictions for Stage1 VLM + baseline Stage2 action expert."""

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
STAGE1_CKPT = Path(
    "/data/alpamayo_sft_artifacts/"
    "output_stage1_nav_smoke_stage1overfit300_20260623_104948/checkpoint-300"
)
DATASET_DIR = Path("/data/datasets/physical_ai_av")
SAMPLES_JSON = Path("/data/alpamayo_sft_artifacts/nav_demo_samples.json")
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
    parser.add_argument(
        "--rows",
        default="all",
        help="'all' for the default 20-row export, or comma-separated row indexes for diagnostics.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-p", type=float, default=0.98)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--diffusion-temperature", type=float, default=0.6)
    parser.add_argument("--num-traj-samples", type=int, default=1)
    parser.add_argument("--num-traj-sets", type=int, default=1)
    parser.add_argument("--max-generation-length", type=int, default=256)
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


def selected_rows(selector: str) -> list[dict[str, Any]]:
    rows = json.loads(SAMPLES_JSON.read_text(encoding="utf-8"))
    if len(rows) != EXPECTED_NAV_DEMO_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_NAV_DEMO_ROWS} nav-demo rows, got {len(rows)}")
    for idx, row in enumerate(rows):
        row["row_index"] = idx
    if selector == "all":
        return rows
    indexes = [int(part) for part in selector.split(",") if part]
    return [rows[index] for index in indexes]


def write_annotation(path: Path, row: dict[str, Any]) -> None:
    payload = [{k: v for k, v in row.items() if k != "row_index"}]
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


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


def summarize(records: list[dict[str, Any]], runtime_s: float, device: torch.device) -> dict[str, Any]:
    def finite_values(key: str) -> list[float]:
        return [r[key] for r in records if r.get(key) is not None and math.isfinite(r[key])]

    ades = finite_values("ade")
    min_ades = finite_values("min_ade")
    corners = finite_values("corner_distance")
    return {
        "model": "stage1_vlm_baseline_expert",
        "num_rows": len(records),
        "mean_ade": float(np.mean(ades)) if ades else None,
        "median_ade": float(np.median(ades)) if ades else None,
        "mean_min_ade": float(np.mean(min_ades)) if min_ades else None,
        "median_min_ade": float(np.median(min_ades)) if min_ades else None,
        "mean_corner_distance": float(np.mean(corners)) if corners else None,
        "median_corner_distance": float(np.median(corners)) if corners else None,
        "runtime_s": round(runtime_s, 3),
        "peak_vram_mib": (
            int(torch.cuda.max_memory_allocated(device) / 1024 / 1024)
            if torch.cuda.is_available()
            else None
        ),
    }


def main() -> None:
    args = parse_args()
    output_root = args.output_root
    protected_roots = {
        Path("/"),
        Path("/data"),
        Path("/data/alpamayo_sft_artifacts"),
        BASE_A1,
        STAGE1_CKPT,
        STAGE1_CKPT.parent,
        DATASET_DIR,
    }
    resolved_output = output_root.resolve()
    resolved_protected = {path.resolve() for path in protected_roots}
    if resolved_output in resolved_protected:
        raise SystemExit(f"Refusing unsafe output root: {output_root}")

    if output_root.exists():
        shutil.rmtree(output_root)
    model_dir = output_root / "stage1_vlm_baseline_expert"
    ann_dir = output_root / "_row_annotations"
    model_dir.mkdir(parents=True)
    ann_dir.mkdir(parents=True)

    device = torch.device(args.device)
    set_seed(args.seed)
    model = TrainableAlpamayoR1.from_pretrained(
        str(BASE_A1),
        stage1_vlm_checkpoint_path=str(STAGE1_CKPT),
    )
    model.to(device)
    model.eval()

    records: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    rows = selected_rows(args.rows)
    start = time.perf_counter()
    for row in rows:
        row_index = int(row["row_index"])
        annotation_path = ann_dir / f"row_{row_index:02d}.json"
        write_annotation(annotation_path, row)
        batch = build_batch(model, annotation_path, device)
        set_seed(args.seed)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            pred_xyz, pred_rot = model.sample_trajectories_from_data(
                data=batch,
                top_p=args.top_p,
                temperature=args.temperature,
                num_traj_samples=args.num_traj_samples,
                num_traj_sets=args.num_traj_sets,
                max_generation_length=args.max_generation_length,
                diffusion_kwargs={"temperature": args.diffusion_temperature},
            )

        metrics = metric_values(pred_xyz, pred_rot, batch)
        prefix = f"row_{row_index:02d}"
        arrays[f"{prefix}/pred_xyz"] = pred_xyz.detach().cpu().numpy()
        arrays[f"{prefix}/pred_rot"] = pred_rot.detach().cpu().numpy()
        arrays[f"{prefix}/ego_future_xyz"] = batch["ego_future_xyz"].detach().cpu().numpy()
        arrays[f"{prefix}/ego_future_rot"] = batch["ego_future_rot"].detach().cpu().numpy()
        arrays[f"{prefix}/ego_history_xyz"] = batch["ego_history_xyz"].detach().cpu().numpy()
        arrays[f"{prefix}/ego_history_rot"] = batch["ego_history_rot"].detach().cpu().numpy()
        records.append(
            {
                "model": "stage1_vlm_baseline_expert",
                "row_index": row_index,
                "clip_id": row["clip_id"],
                "t0_relative": row.get("t0_relative"),
                "nav_text": row["nav_text"],
                "ade": metrics["ade"],
                "min_ade": metrics["min_ade"],
                "corner_distance": metrics.get("corner_distance"),
                "endpoint_xy": arrays[f"{prefix}/pred_xyz"][0, 0, 0, -1, :2].astype(float).tolist(),
                "npz_key_prefix": prefix,
                "seed": args.seed,
                "top_p": args.top_p,
                "temperature": args.temperature,
                "diffusion_temperature": args.diffusion_temperature,
                "num_traj_samples": args.num_traj_samples,
                "num_traj_sets": args.num_traj_sets,
                "max_generation_length": args.max_generation_length,
                "load_contract": (
                    "TrainableAlpamayoR1.from_pretrained(BASE_A1, "
                    "stage1_vlm_checkpoint_path=STAGE1_CKPT); baseline action expert, frozen Stage1 VLM"
                ),
            }
        )

    np.savez_compressed(model_dir / "predictions.npz", **arrays)
    with (model_dir / "results.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = summarize(records, runtime_s=time.perf_counter() - start, device=device)
    (model_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    manifest = {
        "created_for": "stage1_vlm_baseline_expert_discriminating_probe",
        "rows": [int(row["row_index"]) for row in rows],
        "paths": {
            "baseline_a1": str(BASE_A1),
            "stage1_checkpoint": str(STAGE1_CKPT),
            "samples_json": str(SAMPLES_JSON),
        },
        "summary": summary,
        "evaluation_policy": {
            "default_unit": "full_20_rows",
            "row07_note": "row07 is one historical reproducibility reference among 20 rows, not a gate or primary decision row.",
        },
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"summary": summary, "output_root": str(output_root)}, indent=2))


if __name__ == "__main__":
    main()
