#!/usr/bin/env python3
"""Replay validator + score engine on existing pipeline JSON files.

Qwen/OCR/visual outputs are reused unchanged. This is appropriate when a new
version only modifies deterministic post-processing (scene resolution,
subject alignment, evidence validation, or score rules).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.evidence_validator import validate_evidence
from modules.score_engine import calculate_scores

PIPELINE_VERSION = "mvp_ocr_qwen_validator_score_v1.3.7.1"


def safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Directory containing *_result.json")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*_result.json"))
    ok = failed = skipped = 0
    started = time.perf_counter()

    for path in files:
        out_path = output_dir / path.name
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            t0 = time.perf_counter()
            evidence = validate_evidence(
                safe_dict(data.get("ocr")),
                safe_dict(data.get("visual_features")),
                safe_dict(data.get("qwen_analysis")),
                safe_dict(data.get("sam_features")),
                data.get("image_path"),
            )
            validator_seconds = time.perf_counter() - t0
            t1 = time.perf_counter()
            score_result = calculate_scores(evidence)
            score_seconds = time.perf_counter() - t1

            old_version = data.get("pipeline_version", "")
            data["pipeline_version"] = PIPELINE_VERSION
            data["validated_evidence"] = evidence
            data["score_result"] = score_result
            runtime = safe_dict(data.get("runtime"))
            runtime["validator_seconds"] = round(validator_seconds, 4)
            runtime["score_seconds"] = round(score_seconds, 4)
            runtime["replay_seconds"] = round(validator_seconds + score_seconds, 4)
            data["runtime"] = runtime
            data["replay_metadata"] = {
                "replayed": True,
                "source_json": str(path),
                "source_pipeline_version": old_version,
                "target_pipeline_version": PIPELINE_VERSION,
                "replayed_at": datetime.now().isoformat(timespec="seconds"),
                "reused_modules": ["ocr", "visual_features", "qwen_analysis"],
                "recomputed_modules": ["validated_evidence", "score_result"],
            }
            out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            ok += 1
        except Exception as exc:
            failed += 1
            print(f"失败：{path}: {type(exc).__name__}: {exc}")

    summary = {
        "input": str(input_dir),
        "output": str(output_dir),
        "total": len(files),
        "ok": ok,
        "skipped": skipped,
        "failed": failed,
        "seconds": round(time.perf_counter() - started, 4),
        "pipeline_version": PIPELINE_VERSION,
    }
    (output_dir / "replay_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
