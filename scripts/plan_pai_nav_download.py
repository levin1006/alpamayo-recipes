#!/usr/bin/env python3
"""Plan or run the PAI component download command for Alpamayo 1.5 nav SFT.

This is intentionally a thin wrapper around ``scripts/download_pai.py``. It
adds a dry-run default, a stable log path, and a small failure record when the
wrapped command exits non-zero.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path


DEFAULT_NAV_CHUNKS = [
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
DEFAULT_CAMERAS = [
    "camera_front_wide_120fov",
    "camera_cross_left_120fov",
    "camera_cross_right_120fov",
    "camera_front_tele_30fov",
]
DEFAULT_CALIBRATION = ["camera_intrinsics", "sensor_extrinsics"]
DEFAULT_LABELS = ["egomotion"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dry-run or execute the PAI nav component download plan."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="PAI dataset root to pass to scripts/download_pai.py.",
    )
    parser.add_argument(
        "--chunk-ids",
        nargs="+",
        type=int,
        default=DEFAULT_NAV_CHUNKS,
        help="Chunk IDs to download. Defaults to the bundled nav annotation chunks.",
    )
    parser.add_argument("--camera", nargs="+", default=DEFAULT_CAMERAS)
    parser.add_argument("--calibration", nargs="+", default=DEFAULT_CALIBRATION)
    parser.add_argument("--labels", nargs="+", default=DEFAULT_LABELS)
    parser.add_argument(
        "--reasoning",
        nargs="+",
        default=None,
        help="Optional reasoning subparts to pass through, e.g. ood_reasoning.parquet.",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("logs/pai_download"),
        help="Directory for command logs and failure JSON.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually call scripts/download_pai.py. Omit for dry-run/plan mode.",
    )
    return parser.parse_args()


def build_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/download_pai.py",
        "--chunk-ids",
        " ".join(str(chunk_id) for chunk_id in args.chunk_ids),
        "--output-dir",
        str(args.output_dir),
    ]
    if args.camera:
        cmd.extend(["--camera", *args.camera])
    if args.calibration:
        cmd.extend(["--calibration", *args.calibration])
    if args.labels:
        cmd.extend(["--labels", *args.labels])
    if args.reasoning:
        cmd.extend(["--reasoning", *args.reasoning])
    return cmd


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    command = build_command(args)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.logs_dir / f"pai_nav_download_{timestamp}.log"
    failure_path = args.logs_dir / f"pai_nav_download_failed_{timestamp}.json"

    printable = " ".join(shlex.quote(part) for part in command)
    print("[plan_pai_nav_download] Command:")
    print(printable)
    print(f"[plan_pai_nav_download] Log path: {log_path}")

    if not args.execute:
        print("[plan_pai_nav_download] Dry-run only. Add --execute to download.")
        log_path.write_text(printable + "\n", encoding="utf-8")
        return 0

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(printable + "\n\n")
        log_file.flush()
        process = subprocess.Popen(
            command,
            cwd=repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            log_file.flush()
        returncode = process.wait()

    if returncode != 0:
        failure = {
            "returncode": returncode,
            "command": command,
            "output_dir": str(args.output_dir),
            "chunk_ids": args.chunk_ids,
            "camera": args.camera,
            "calibration": args.calibration,
            "labels": args.labels,
            "reasoning": args.reasoning,
            "log_path": str(log_path),
        }
        failure_path.write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
        print(f"[plan_pai_nav_download] Download failed. Failure JSON: {failure_path}")
        return returncode

    print("[plan_pai_nav_download] Download command completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
