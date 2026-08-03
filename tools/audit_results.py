#!/usr/bin/env python3
"""Export a compact CSV audit table from pipeline JSON results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def join_values(value: Any) -> str:
    if isinstance(value, list):
        return " | ".join(str(item) for item in value)
    return str(value or "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/json_results")
    parser.add_argument("--output", default="outputs/v13_audit.csv")
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = []
    for path in sorted(Path(args.input).glob("*_result.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            rows.append({"json_file": str(path), "parse_error": f"{type(exc).__name__}: {exc}"})
            continue

        evidence = safe_dict(data.get("validated_evidence"))
        qwen = safe_dict(data.get("qwen_analysis"))
        scores = safe_dict(safe_dict(data.get("score_result")).get("scores"))
        text_analysis = safe_dict(evidence.get("text_analysis"))
        subject_alignment = safe_dict(evidence.get("subject_alignment"))
        subject_resolution = safe_dict(evidence.get("subject_resolution"))
        brand_info = safe_dict(evidence.get("brand_info"))
        text_roles = safe_dict(evidence.get("text_roles"))
        qwen_runtime = safe_dict(qwen.get("_runtime"))
        rows.append({
            "result_id": data.get("result_id", ""),
            "image_path": data.get("image_path", ""),
            "image_sha256": data.get("image_sha256", ""),
            "pipeline_version": data.get("pipeline_version", ""),
            "scene_type": evidence.get("scene_type", ""),
            "main_subject": evidence.get("main_subject", ""),
            "visual_subject": evidence.get("visual_subject", ""),
            "advertised_subject": evidence.get("advertised_subject", ""),
            "product_type": evidence.get("product_type", ""),
            "visual_subject_changed": bool(subject_resolution.get("changed")),
            "visual_subject_source": subject_resolution.get("source", ""),
            "visual_subject_confidence": subject_resolution.get("confidence", ""),
            "format_resolution_changed": bool(safe_dict(evidence.get("format_resolution")).get("changed")),
            "format_resolution_reasons": join_values(safe_dict(evidence.get("format_resolution")).get("reasons")),
            "face_count": safe_dict(subject_resolution.get("face_evidence")).get("face_count", ""),
            "subject_alignment_changed": bool(subject_alignment.get("changed")),
            "subject_alignment_profile": subject_alignment.get("profile", ""),
            "subject_alignment_confidence": subject_alignment.get("confidence", ""),
            "text_status": text_analysis.get("status", ""),
            "suspicious_text": join_values(text_analysis.get("suspicious_lines")),
            "brand": brand_info.get("brand_text", ""),
            "brand_type": brand_info.get("brand_type", ""),
            "brand_status": brand_info.get("brand_status", ""),
            "brand_candidate": brand_info.get("brand_candidate", ""),
            "brand_confidence": brand_info.get("confidence", ""),
            "brand_candidates": join_values([
                f"{item.get('text', '')}[{item.get('status', '')}]"
                for item in text_roles.get("brand_candidates", []) if isinstance(item, dict)
            ]),
            "direct_cta": join_values(text_roles.get("direct_cta_text")),
            "direct_cta_raw": join_values(text_roles.get("direct_cta_raw_text")),
            "direct_cta_canonical": join_values(text_roles.get("direct_cta_canonical")),
            "contextual_cta": join_values(text_roles.get("contextual_cta_text")),
            "contextual_cta_raw": join_values(text_roles.get("contextual_cta_raw_text")),
            "contextual_cta_canonical": join_values(text_roles.get("contextual_cta_canonical")),
            "price": safe_dict(evidence.get("price_info")).get("price_text", ""),
            "promotion": join_values(safe_dict(evidence.get("promotion_info")).get("promotion_words")),
            "cta": safe_dict(evidence.get("cta_info")).get("cta_text", ""),
            "cta_raw": safe_dict(evidence.get("cta_info")).get("cta_raw_text", ""),
            "cta_canonical": join_values(safe_dict(evidence.get("cta_info")).get("cta_canonical")),
            "event_time": safe_dict(evidence.get("event_info")).get("event_time", ""),
            "event_location": safe_dict(evidence.get("event_info")).get("event_location", ""),
            "risk_points": join_values(evidence.get("risk_points")),
            "fallback": bool(qwen.get("fallback")),
            "json_retry": bool(qwen_runtime.get("json_retry")),
            "protocol_complete": bool(qwen_runtime.get("protocol_complete")),
            "qwen_parse_error": bool(qwen_runtime.get("parse_error")),
            "persuasion": scores.get("persuasion", ""),
            "arousal": scores.get("arousal", ""),
            "trust": scores.get("trust", ""),
            "attention": scores.get("attention", ""),
            "memory": scores.get("memory", ""),
            "total": scores.get("total", ""),
            "runtime_seconds": safe_dict(data.get("runtime")).get("total_seconds", ""),
            "json_file": str(path),
            "parse_error": "",
        })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["json_file", "parse_error"]
    with output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"已导出 {len(rows)} 条：{output}")


if __name__ == "__main__":
    main()
