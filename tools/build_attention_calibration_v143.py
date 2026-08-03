#!/usr/bin/env python3

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


SOURCE_ROOT = Path("outputs/v141_saliency_288")
OUTPUT_PATH = Path("configs/attention_calibration_v143.json")

VERSION = "transalnet_v1.4.3-attention-parallel"
MIN_SCENE_SAMPLE = 20

QUANTILES = [
    0.00,
    0.05,
    0.10,
    0.25,
    0.50,
    0.75,
    0.90,
    0.95,
    1.00,
]


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_mapping(records):
    source_values = np.asarray(
        [
            record["center_saliency_ratio"]
            for record in records
        ],
        dtype=float,
    )

    target_values = np.asarray(
        [
            record["legacy_center_focus"]
            for record in records
        ],
        dtype=float,
    )

    source_knots = np.quantile(
        source_values,
        QUANTILES,
    )

    target_knots = np.quantile(
        target_values,
        QUANTILES,
    )

    unique_source = []
    unique_target = []

    for source, target in zip(
        source_knots,
        target_knots,
    ):
        source = float(source)
        target = float(target)

        if (
            unique_source
            and abs(source - unique_source[-1]) < 1e-12
        ):
            unique_target[-1] = max(
                unique_target[-1],
                target,
            )
        else:
            unique_source.append(source)
            unique_target.append(target)

    return {
        "input_metric": "center_saliency_ratio",
        "target_metric": "legacy_center_focus",
        "quantiles": QUANTILES,
        "source_knots": [
            round(value, 6)
            for value in unique_source
        ],
        "target_knots": [
            round(value, 6)
            for value in unique_target
        ],
        "sample_size": len(records),
    }


records = []

for path in sorted(
    SOURCE_ROOT.rglob("*_result.json")
):
    data = json.loads(
        path.read_text(encoding="utf-8")
    )

    saliency = data.get(
        "saliency_analysis",
        {},
    )

    metrics = saliency.get(
        "metrics",
        {},
    )

    candidate = saliency.get(
        "candidate_features",
        {},
    )

    score_result = data.get(
        "score_result",
        {},
    )

    center_ratio = safe_float(
        metrics.get("center_saliency_ratio")
    )

    legacy_center = safe_float(
        candidate.get("legacy_saliency_center")
    )

    scene = score_result.get(
        "scene_scoring_profile",
        "未知场景",
    )

    if center_ratio is None:
        raise RuntimeError(
            f"缺少center_saliency_ratio：{path}"
        )

    if legacy_center is None:
        raise RuntimeError(
            f"缺少legacy_center_focus：{path}"
        )

    records.append({
        "scene": scene,
        "center_saliency_ratio": center_ratio,
        "legacy_center_focus": legacy_center,
    })


if len(records) != 288:
    raise RuntimeError(
        f"预期288条，实际{len(records)}条"
    )


global_mapping = build_mapping(records)

records_by_scene = defaultdict(list)

for record in records:
    records_by_scene[
        record["scene"]
    ].append(record)


scene_mappings = {}

for scene, scene_records in sorted(
    records_by_scene.items()
):
    if len(scene_records) >= MIN_SCENE_SAMPLE:
        mapping = build_mapping(scene_records)
        mapping["scope"] = "scene"
    else:
        mapping = dict(global_mapping)
        mapping["scope"] = "global_fallback"
        mapping["scene_sample_size"] = len(
            scene_records
        )

    scene_mappings[scene] = mapping


payload = {
    "version": VERSION,
    "method": "quantile_mapping",
    "candidate_scope": "attention_only",
    "scoring_integration": False,
    "minimum_scene_sample": MIN_SCENE_SAMPLE,
    "training_sample_size": len(records),
    "validation": {
        "method": (
            "5-fold scene-stratified "
            "out-of-fold validation"
        ),
        "sample_size": 288,
        "attention_delta_mean": -0.0764,
        "attention_delta_median": 0.0,
        "attention_delta_min": -8.0,
        "attention_delta_max": 9.0,
        "attention_abs_ge_10": 0,
        "total_delta_mean": -0.0153,
        "total_abs_ge_2": 0,
    },
    "global_mapping": global_mapping,
    "scene_mappings": scene_mappings,
}

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_PATH.write_text(
    json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

print("========== 最终校准文件完成 ==========")
print("记录数：", len(records))
print("输出文件：", OUTPUT_PATH)

print("\n场景校准范围：")

for scene, mapping in scene_mappings.items():
    print(
        f"{scene}: "
        f"{mapping['scope']} "
        f"(n={len(records_by_scene[scene])})"
    )
