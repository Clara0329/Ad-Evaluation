#!/usr/bin/env python3
"""Apply the v1.3.7.1 CTA text-preservation patch to existing v1.3.7 JSON.

This replay is deliberately field-scoped: it recomputes only OCR text roles
needed for CTA display/evidence and leaves scene, subjects, brand, event facts,
score_result and all non-CTA evidence unchanged.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.text_role_classifier import classify_text_roles

PIPELINE_VERSION = "mvp_ocr_qwen_validator_score_v1.3.7.1"
CTA_ROLE_FIELDS = (
    "cta_text",
    "cta_raw_text",
    "cta_canonical",
    "cta_evidence",
    "direct_cta_text",
    "direct_cta_raw_text",
    "direct_cta_canonical",
    "direct_cta_evidence",
    "contextual_cta_text",
    "contextual_cta_raw_text",
    "contextual_cta_canonical",
    "contextual_cta_evidence",
)


def safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def merge_cta_roles(old_roles: Dict[str, Any], new_roles: Dict[str, Any]) -> Dict[str, Any]:
    """Keep every non-CTA role field unchanged and replace CTA-specific fields."""
    result = copy.deepcopy(old_roles) if old_roles else copy.deepcopy(new_roles)
    for field in CTA_ROLE_FIELDS:
        result[field] = copy.deepcopy(new_roles.get(field, []))

    old_records = result.get("roles") if isinstance(result.get("roles"), list) else []
    new_records = new_roles.get("roles") if isinstance(new_roles.get("roles"), list) else []
    new_by_index = {
        item.get("index"): item
        for item in new_records
        if isinstance(item, dict) and item.get("index") is not None
    }
    merged_records: List[Dict[str, Any]] = []
    for item in old_records:
        if not isinstance(item, dict):
            merged_records.append(item)
            continue
        merged = copy.deepcopy(item)
        replacement = new_by_index.get(item.get("index"))
        if replacement and (item.get("role") == "cta" or replacement.get("role") == "cta"):
            merged["role"] = replacement.get("role", merged.get("role"))
            merged["reason"] = replacement.get("reason", merged.get("reason"))
        merged_records.append(merged)
    if old_records:
        result["roles"] = merged_records
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Existing v1.3.7 *_result.json directory")
    parser.add_argument("--output", required=True, help="v1.3.7.1 output directory")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(input_dir.glob("*_result.json"))

    ok = failed = skipped = changed = 0
    presence_changes: List[str] = []
    started = time.perf_counter()

    for path in files:
        out_path = output_dir / path.name
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            t0 = time.perf_counter()
            evidence = safe_dict(data.get("validated_evidence"))
            old_cta = safe_dict(evidence.get("cta_info"))
            old_has_cta = bool(old_cta.get("has_cta"))

            visual = safe_dict(data.get("visual_features"))
            image_size = safe_dict(visual.get("image_size"))
            new_roles = classify_text_roles(
                safe_dict(data.get("ocr")),
                str(evidence.get("scene_type", "") or ""),
                image_size,
            )
            ctas = list(new_roles.get("cta_text") or [])
            raw_ctas = list(new_roles.get("cta_raw_text") or [])
            canonical = list(new_roles.get("cta_canonical") or [])
            cta_evidence = [
                item for item in (new_roles.get("cta_evidence") or [])
                if isinstance(item, dict)
            ]
            new_has_cta = bool(ctas)
            image_id = Path(str(data.get("image_path", "") or path.stem)).stem
            if old_has_cta != new_has_cta:
                presence_changes.append(image_id)
                raise ValueError(
                    f"CTA存在性发生变化：{image_id}: {old_has_cta} -> {new_has_cta}。"
                    "v1.3.7.1仅允许原文保留修复。"
                )

            old_text = str(old_cta.get("cta_text", "") or "")
            new_text = "，".join(ctas)
            if old_text != new_text:
                changed += 1

            old_role_data = safe_dict(evidence.get("text_roles"))
            evidence["text_roles"] = merge_cta_roles(old_role_data, new_roles)
            new_cta_info = copy.deepcopy(old_cta)
            new_cta_info.update({
                "has_cta": new_has_cta,
                "cta_text": new_text,
                "cta_raw_text": "，".join(raw_ctas),
                "cta_canonical": canonical,
                "evidence_source": "ocr" if new_has_cta else "",
                "confidence": 0.96 if new_has_cta and raw_ctas else (0.9 if new_has_cta else 0.0),
                "evidence": cta_evidence,
            })
            evidence["cta_info"] = new_cta_info

            provenance = safe_dict(evidence.get("field_provenance"))
            provenance["cta_info"] = {
                "value": ctas,
                "sources": ["ocr", "text_role_rules"] if new_has_cta else [],
                "inferred": False,
                "evidence": cta_evidence or raw_ctas or ctas,
            }
            evidence["field_provenance"] = provenance

            data["validated_evidence"] = evidence
            old_version = str(data.get("pipeline_version", "") or "")
            data["pipeline_version"] = PIPELINE_VERSION
            runtime = safe_dict(data.get("runtime"))
            runtime["cta_patch_seconds"] = round(time.perf_counter() - t0, 4)
            data["runtime"] = runtime
            data["replay_metadata"] = {
                "replayed": True,
                "patch_scope": "cta_text_preservation_only",
                "source_json": str(path),
                "source_pipeline_version": old_version,
                "target_pipeline_version": PIPELINE_VERSION,
                "replayed_at": datetime.now().isoformat(timespec="seconds"),
                "reused_modules": [
                    "ocr", "visual_features", "qwen_analysis", "scene", "subjects",
                    "brand", "event_facts", "score_result"
                ],
                "recomputed_modules": ["cta_text_roles", "cta_info"],
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
        "cta_text_changed": changed,
        "cta_presence_changed": presence_changes,
        "seconds": round(time.perf_counter() - started, 4),
        "pipeline_version": PIPELINE_VERSION,
        "patch_scope": "cta_text_preservation_only",
    }
    (output_dir / "replay_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
