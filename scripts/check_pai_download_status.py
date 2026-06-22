#!/usr/bin/env python3
"""Read-only PAI dataset status and nav annotation coverage report."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_COMPONENTS = [
    "camera_front_wide_120fov",
    "camera_cross_left_120fov",
    "camera_cross_right_120fov",
    "camera_front_tele_30fov",
    "camera_intrinsics",
    "sensor_extrinsics",
    "egomotion",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check local PAI dataset status.")
    parser.add_argument(
        "--roots",
        nargs="+",
        type=Path,
        required=True,
        help="One or more candidate PAI dataset roots.",
    )
    parser.add_argument(
        "--components",
        nargs="+",
        default=DEFAULT_COMPONENTS,
        help="Feature/component names to check. Defaults to Alpamayo 1.5 nav components.",
    )
    parser.add_argument(
        "--chunk-ids",
        nargs="+",
        type=int,
        default=None,
        help="Optional chunk IDs to restrict component inventory and clip catalog outputs.",
    )
    parser.add_argument(
        "--nav-annotations",
        type=Path,
        default=None,
        help="Optional nav annotation JSON, e.g. nav_demo_samples.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs/pai_inventory"),
        help="Directory where CSV/JSON/Markdown reports are written.",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Output filename prefix. Defaults to pai_status_<timestamp>.",
    )
    return parser.parse_args()


def safe_root_name(root: Path) -> str:
    return str(root.resolve()).strip("/").replace("/", "__")


def expected_chunk_path(root: Path, chunk_path_template: str, chunk_id: int) -> Path:
    return root / chunk_path_template.format(chunk_id=int(chunk_id))


def load_required_metadata(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features_path = root / "features.csv"
    clip_index_path = root / "clip_index.parquet"
    feature_presence_path = root / "metadata" / "feature_presence.parquet"
    missing = [
        str(path)
        for path in [features_path, clip_index_path, feature_presence_path]
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing required PAI metadata: " + ", ".join(missing))
    return (
        pd.read_csv(features_path),
        pd.read_parquet(clip_index_path),
        pd.read_parquet(feature_presence_path),
    )


def clip_catalog(clip_index: pd.DataFrame) -> pd.DataFrame:
    out = clip_index.reset_index()
    if "index" in out.columns and "clip_id" not in out.columns:
        out = out.rename(columns={"index": "clip_id"})
    preferred = [
        col
        for col in [
            "clip_id",
            "chunk",
            "split",
            "clip_is_valid",
            "event_t0s",
            "t0",
            "t0_relative",
            "keyframe",
            "key_frame",
        ]
        if col in out.columns
    ]
    return out[preferred] if preferred else out


def add_nav_columns(catalog: pd.DataFrame, nav_df: pd.DataFrame | None) -> pd.DataFrame:
    """Add clip-level nav annotation coverage columns to a clip catalog."""
    if nav_df is None or "clip_id" not in catalog.columns:
        return catalog

    required = {"clip_id", "t0_relative", "nav_text"}
    if not required.issubset(nav_df.columns):
        out = catalog.copy()
        out["nav_annotation_count"] = 0
        out["has_nav_annotation"] = False
        return out

    grouped = (
        nav_df.assign(
            t0_relative_str=nav_df["t0_relative"].astype(str),
            nav_text_str=nav_df["nav_text"].astype(str),
        )
        .groupby("clip_id")
        .agg(
            nav_annotation_count=("clip_id", "size"),
            nav_t0_relative_first=("t0_relative", "first"),
            nav_text_first=("nav_text", "first"),
            nav_t0_relative_list=("t0_relative_str", lambda values: ";".join(values)),
            nav_text_list=("nav_text_str", lambda values: " || ".join(values)),
        )
        .reset_index()
    )
    out = catalog.merge(grouped, on="clip_id", how="left")
    out["nav_annotation_count"] = out["nav_annotation_count"].fillna(0).astype(int)
    out["has_nav_annotation"] = out["nav_annotation_count"] > 0
    return out


def restrict_clip_index(clip_index: pd.DataFrame, chunk_ids: list[int] | None) -> pd.DataFrame:
    if chunk_ids is None:
        return clip_index
    allowed = set(int(chunk_id) for chunk_id in chunk_ids)
    return clip_index.loc[clip_index["chunk"].isin(allowed)]


def component_inventory(
    root: Path,
    features: pd.DataFrame,
    clip_index: pd.DataFrame,
    feature_presence: pd.DataFrame,
    components: list[str],
    chunk_ids: list[int] | None,
) -> pd.DataFrame:
    feature_by_name = features.set_index("feature", drop=False)
    if chunk_ids is None:
        chunks = sorted(int(chunk) for chunk in clip_index["chunk"].dropna().unique())
    else:
        chunks = sorted(set(int(chunk_id) for chunk_id in chunk_ids))
    rows: list[dict[str, Any]] = []

    presence_with_chunk = pd.concat(
        [clip_index[["chunk"]].reset_index(drop=True), feature_presence.reset_index(drop=True)],
        axis=1,
    )

    for component in components:
        if component not in feature_by_name.index or component not in feature_presence.columns:
            rows.append(
                {
                    "root": str(root),
                    "component": component,
                    "chunk": "",
                    "status": "metadata_absent",
                    "expected_path": "",
                    "metadata_clip_count": 0,
                    "file_exists": False,
                    "file_size_bytes": 0,
                }
            )
            continue

        feature_row = feature_by_name.loc[component]
        chunk_path_template = str(feature_row["chunk_path"])
        grouped_presence = presence_with_chunk.groupby("chunk")[component].sum()
        for chunk_id in chunks:
            metadata_clip_count = int(grouped_presence.get(chunk_id, 0))
            expected_path = expected_chunk_path(root, chunk_path_template, chunk_id)
            exists = expected_path.is_file()
            if exists:
                status = "downloaded"
            elif metadata_clip_count > 0:
                status = "metadata_present_file_missing"
            else:
                status = "metadata_present_not_required_for_chunk"
            rows.append(
                {
                    "root": str(root),
                    "component": component,
                    "chunk": chunk_id,
                    "status": status,
                    "expected_path": str(expected_path),
                    "metadata_clip_count": metadata_clip_count,
                    "file_exists": exists,
                    "file_size_bytes": expected_path.stat().st_size if exists else 0,
                }
            )

    return pd.DataFrame(rows)


def summarize_root(root: Path, features: pd.DataFrame, clip_index: pd.DataFrame) -> dict[str, Any]:
    chunks = clip_index["chunk"] if "chunk" in clip_index.columns else pd.Series(dtype=int)
    return {
        "root": str(root),
        "resolved_root": str(root.resolve()),
        "features_csv": str(root / "features.csv"),
        "clip_index_parquet": str(root / "clip_index.parquet"),
        "feature_presence_parquet": str(root / "metadata" / "feature_presence.parquet"),
        "feature_count": int(len(features)),
        "clip_count": int(len(clip_index)),
        "chunk_count": int(chunks.nunique()) if len(chunks) else 0,
        "chunk_min": int(chunks.min()) if len(chunks) else None,
        "chunk_max": int(chunks.max()) if len(chunks) else None,
    }


def load_nav_annotations(path: Path) -> pd.DataFrame:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected list JSON in {path}")
    return pd.DataFrame(rows)


def nav_join_report(root: Path, clip_index: pd.DataFrame, nav_df: pd.DataFrame) -> dict[str, Any]:
    clip_index_reset = clip_index.reset_index()
    if "index" in clip_index_reset.columns and "clip_id" not in clip_index_reset.columns:
        clip_index_reset = clip_index_reset.rename(columns={"index": "clip_id"})
    required = {"clip_id", "t0_relative", "nav_text"}
    missing_schema = sorted(required - set(nav_df.columns))
    if missing_schema:
        return {
            "root": str(root),
            "schema_ok": False,
            "missing_schema": missing_schema,
            "annotation_rows": int(len(nav_df)),
            "matched_rows": 0,
            "matched_chunks": [],
        }
    merged = nav_df.merge(clip_index_reset[["clip_id", "chunk"]], on="clip_id", how="left")
    matched = merged["chunk"].notna()
    chunks = sorted(int(chunk) for chunk in merged.loc[matched, "chunk"].unique())
    return {
        "root": str(root),
        "schema_ok": True,
        "missing_schema": [],
        "annotation_rows": int(len(nav_df)),
        "matched_rows": int(matched.sum()),
        "missing_rows": int((~matched).sum()),
        "matched_chunks": chunks,
    }


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# PAI Download Status",
        "",
        f"Generated at: {summary['generated_at']}",
        "",
        "## Roots",
        "",
        "| Root | Clips | Chunks | Features | Chunk range |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for root in summary["roots"]:
        lines.append(
            f"| `{root['root']}` | {root['clip_count']} | {root['chunk_count']} | "
            f"{root['feature_count']} | {root['chunk_min']}..{root['chunk_max']} |"
        )

    lines.extend(["", "## Component Status Summary", ""])
    lines.append("| Root | Component | Downloaded | Missing file | Metadata absent |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for row in summary["component_summary"]:
        lines.append(
            f"| `{row['root']}` | `{row['component']}` | {row.get('downloaded', 0)} | "
            f"{row.get('metadata_present_file_missing', 0)} | {row.get('metadata_absent', 0)} |"
        )

    if summary.get("nav_annotations"):
        lines.extend(["", "## Nav Annotation Join", ""])
        lines.append(
            f"Annotation rows: {summary['nav_annotations']['rows']}; "
            f"unique annotated clips: {summary['nav_annotations']['unique_clip_count']}"
        )
        lines.append("")
        lines.append("| Root | Schema OK | Rows | Matched | Missing | Matched chunks |")
        lines.append("| --- | --- | ---: | ---: | ---: | --- |")
        for row in summary["nav_annotations"]["joins"]:
            lines.append(
                f"| `{row['root']}` | {row['schema_ok']} | {row['annotation_rows']} | "
                f"{row['matched_rows']} | {row.get('missing_rows', 0)} | "
                f"`{row.get('matched_chunks', [])}` |"
            )
        if summary["nav_annotations"].get("coverage"):
            lines.extend(["", "## Clip-Level Nav Coverage", ""])
            lines.append("| Root | Catalog clips | Clips with nav | Clips without nav |")
            lines.append("| --- | ---: | ---: | ---: |")
            for row in summary["nav_annotations"]["coverage"]:
                lines.append(
                    f"| `{row['root']}` | {row['catalog_clips']} | "
                    f"{row['clips_with_nav_annotation']} | "
                    f"{row['clips_without_nav_annotation']} |"
                )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = args.prefix or f"pai_status_{timestamp}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "chunk_ids_filter": args.chunk_ids,
        "roots": [],
        "component_summary": [],
        "outputs": {},
    }
    all_inventory: list[pd.DataFrame] = []
    nav_df = load_nav_annotations(args.nav_annotations) if args.nav_annotations else None
    nav_joins: list[dict[str, Any]] = []
    nav_coverage: list[dict[str, Any]] = []

    for root in args.roots:
        features, clip_index, feature_presence = load_required_metadata(root)
        filtered_clip_index = restrict_clip_index(clip_index, args.chunk_ids)
        summary["roots"].append(summarize_root(root, features, filtered_clip_index))

        root_name = safe_root_name(root)
        clip_path = args.output_dir / f"{prefix}__{root_name}__clips.csv"
        catalog = add_nav_columns(clip_catalog(filtered_clip_index), nav_df)
        catalog.to_csv(clip_path, index=False)

        inventory = component_inventory(
            root,
            features,
            clip_index,
            feature_presence,
            args.components,
            args.chunk_ids,
        )
        inventory_path = args.output_dir / f"{prefix}__{root_name}__components.csv"
        inventory.to_csv(inventory_path, index=False)
        all_inventory.append(inventory)
        summary["outputs"][str(root)] = {
            "clip_catalog_csv": str(clip_path),
            "component_inventory_csv": str(inventory_path),
        }

        if nav_df is not None:
            nav_joins.append(nav_join_report(root, clip_index, nav_df))
            annotated = catalog.loc[catalog["has_nav_annotation"]]
            without_nav = catalog.loc[~catalog["has_nav_annotation"]]
            annotated_path = (
                args.output_dir / f"{prefix}__{root_name}__clips_with_nav_annotations.csv"
            )
            without_nav_path = (
                args.output_dir / f"{prefix}__{root_name}__clips_without_nav_annotations.csv"
            )
            annotated.to_csv(annotated_path, index=False)
            without_nav.to_csv(without_nav_path, index=False)
            summary["outputs"][str(root)].update(
                {
                    "clips_with_nav_annotations_csv": str(annotated_path),
                    "clips_without_nav_annotations_csv": str(without_nav_path),
                }
            )
            nav_coverage.append(
                {
                    "root": str(root),
                    "catalog_clips": int(len(catalog)),
                    "clips_with_nav_annotation": int(len(annotated)),
                    "clips_without_nav_annotation": int(len(without_nav)),
                }
            )

    if all_inventory:
        combined = pd.concat(all_inventory, ignore_index=True)
        combined_path = args.output_dir / f"{prefix}__components_all_roots.csv"
        combined.to_csv(combined_path, index=False)
        component_summary = (
            combined.groupby(["root", "component", "status"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
            .to_dict(orient="records")
        )
        summary["component_summary"] = component_summary
        summary["outputs"]["combined_component_inventory_csv"] = str(combined_path)

    if nav_df is not None:
        nav_path = args.output_dir / f"{prefix}__nav_annotations.csv"
        nav_df.to_csv(nav_path, index=False)
        summary["nav_annotations"] = {
            "path": str(args.nav_annotations),
            "rows": int(len(nav_df)),
            "unique_clip_count": int(nav_df["clip_id"].nunique()) if "clip_id" in nav_df else 0,
            "columns": list(nav_df.columns),
            "joins": nav_joins,
            "coverage": nav_coverage,
            "csv": str(nav_path),
        }

    summary_path = args.output_dir / f"{prefix}__summary.json"
    markdown_path = args.output_dir / f"{prefix}__summary.md"
    summary["outputs"]["summary_json"] = str(summary_path)
    summary["outputs"]["summary_markdown"] = str(markdown_path)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_markdown(summary, markdown_path)

    print(f"Wrote summary: {summary_path}")
    print(f"Wrote markdown: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
