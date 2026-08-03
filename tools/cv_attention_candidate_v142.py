#!/usr/bin/env python3

import copy
import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from modules.score_engine import calculate_scores


SOURCE_ROOT = Path("outputs/v141_saliency_288")
OUTPUT_ROOT = Path("evaluation_saliency_v142_cv")

DETAIL_CSV = OUTPUT_ROOT / "attention_candidate_v142_cv_detail.csv"
SUMMARY_JSON = OUTPUT_ROOT / "attention_candidate_v142_cv_summary.json"
FOLD_CSV = OUTPUT_ROOT / "attention_candidate_v142_cv_by_fold.csv"
SCENE_CSV = OUTPUT_ROOT / "attention_candidate_v142_cv_by_scene.csv"

FOLD_COUNT = 5
RANDOM_SEED = 20260803
MIN_SCENE_TRAIN = 12

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


def describe(values):
    values = [
        float(value)
        for value in values
        if value is not None
    ]

    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
        }

    return {
        "count": len(values),
        "mean": round(statistics.mean(values), 4),
        "median": round(statistics.median(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def parse_model_topic(image_path):
    parts = Path(image_path).parts

    try:
        index = parts.index("AIGC_images")
        return parts[index + 1], parts[index + 2]
    except (ValueError, IndexError):
        return "", ""


def build_mapping(records):
    raw_values = np.asarray(
        [record["center_saliency_ratio"] for record in records],
        dtype=float,
    )

    target_values = np.asarray(
        [record["legacy_center_focus"] for record in records],
        dtype=float,
    )

    x_quantiles = np.quantile(raw_values, QUANTILES)
    y_quantiles = np.quantile(target_values, QUANTILES)

    x_knots = []
    y_knots = []

    for x_value, y_value in zip(
        x_quantiles,
        y_quantiles,
    ):
        x_value = float(x_value)
        y_value = float(y_value)

        if (
            x_knots
            and abs(x_value - x_knots[-1]) < 1e-12
        ):
            y_knots[-1] = max(
                y_knots[-1],
                y_value,
            )
        else:
            x_knots.append(x_value)
            y_knots.append(y_value)

    return {
        "x": x_knots,
        "y": y_knots,
        "sample_size": len(records),
    }


def apply_mapping(value, mapping):
    x_values = mapping["x"]
    y_values = mapping["y"]

    if not x_values:
        return None

    if len(x_values) == 1:
        return round(
            max(0.0, min(1.0, y_values[0])),
            4,
        )

    mapped = np.interp(
        float(value),
        np.asarray(x_values, dtype=float),
        np.asarray(y_values, dtype=float),
    )

    return round(
        max(0.0, min(1.0, float(mapped))),
        4,
    )


records = []

for json_path in sorted(
    SOURCE_ROOT.rglob("*_result.json")
):
    data = json.loads(
        json_path.read_text(encoding="utf-8")
    )

    saliency = data.get("saliency_analysis", {})
    metrics = saliency.get("metrics", {})
    candidate = saliency.get("candidate_features", {})

    center_ratio = safe_float(
        metrics.get("center_saliency_ratio")
    )

    legacy_center = safe_float(
        candidate.get("legacy_saliency_center")
    )

    evidence = data.get("validated_evidence")
    scores = (
        data.get("score_result", {})
        .get("scores", {})
    )

    scene = (
        data.get("score_result", {})
        .get("scene_scoring_profile", "未知场景")
    )

    image_path = data.get("image_path", "")
    model, topic = parse_model_topic(image_path)

    if center_ratio is None:
        raise RuntimeError(
            f"缺少center_saliency_ratio：{json_path}"
        )

    if legacy_center is None:
        raise RuntimeError(
            f"缺少legacy_center_focus：{json_path}"
        )

    if not isinstance(evidence, dict):
        raise RuntimeError(
            f"缺少validated_evidence：{json_path}"
        )

    records.append({
        "json_path": str(json_path),
        "image_path": image_path,
        "image_id": Path(image_path).stem,
        "model": model,
        "topic": topic,
        "scene": scene,
        "center_saliency_ratio": center_ratio,
        "legacy_center_focus": legacy_center,
        "evidence": evidence,
        "scores": scores,
    })


if len(records) != 288:
    raise RuntimeError(
        f"预期288张，实际{len(records)}张"
    )


# 按场景分层划分5折
groups = defaultdict(list)

for record in records:
    groups[record["scene"]].append(record)

rng = random.Random(RANDOM_SEED)

for scene_records in groups.values():
    rng.shuffle(scene_records)

    for index, record in enumerate(scene_records):
        record["fold"] = index % FOLD_COUNT


detail_rows = []

for fold in range(FOLD_COUNT):
    train_records = [
        record
        for record in records
        if record["fold"] != fold
    ]

    test_records = [
        record
        for record in records
        if record["fold"] == fold
    ]

    global_mapping = build_mapping(train_records)

    train_by_scene = defaultdict(list)

    for record in train_records:
        train_by_scene[
            record["scene"]
        ].append(record)

    scene_mappings = {}

    for scene in groups:
        scene_train = train_by_scene.get(scene, [])

        unique_values = {
            record["center_saliency_ratio"]
            for record in scene_train
        }

        if (
            len(scene_train) >= MIN_SCENE_TRAIN
            and len(unique_values) >= 3
        ):
            scene_mappings[scene] = {
                "scope": "scene",
                **build_mapping(scene_train),
            }
        else:
            scene_mappings[scene] = {
                "scope": "global_fallback",
                **global_mapping,
            }

    for record in test_records:
        mapping = scene_mappings[record["scene"]]

        calibrated_center = apply_mapping(
            record["center_saliency_ratio"],
            mapping,
        )

        candidate_evidence = copy.deepcopy(
            record["evidence"]
        )

        candidate_evidence.setdefault(
            "visual_features",
            {},
        )["center_focus"] = calibrated_center

        candidate_result = calculate_scores(
            candidate_evidence
        )

        candidate_attention = safe_float(
            candidate_result["scores"]["attention"]
        )

        old_attention = safe_float(
            record["scores"]["attention"]
        )

        attention_delta = round(
            candidate_attention - old_attention,
            2,
        )

        candidate_total = round(
            (
                safe_float(record["scores"]["persuasion"])
                + safe_float(record["scores"]["arousal"])
                + safe_float(record["scores"]["trust"])
                + candidate_attention
                + safe_float(record["scores"]["memory"])
            ) / 5,
            2,
        )

        old_total = safe_float(
            record["scores"]["total"]
        )

        total_delta = round(
            candidate_total - old_total,
            2,
        )

        detail_rows.append({
            "fold": fold + 1,
            "生成模型": record["model"],
            "主题": record["topic"],
            "图片编号": record["image_id"],
            "场景评分类型": record["scene"],
            "校准范围": mapping["scope"],
            "训练样本数": mapping["sample_size"],
            "center_saliency_ratio":
                record["center_saliency_ratio"],
            "旧center_focus":
                record["legacy_center_focus"],
            "交叉验证校准值":
                calibrated_center,
            "旧attention":
                old_attention,
            "候选attention":
                candidate_attention,
            "attention变化":
                attention_delta,
            "旧total":
                old_total,
            "候选total":
                candidate_total,
            "total变化":
                total_delta,
            "源JSON":
                record["json_path"],
        })


OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

detail_headers = [
    "fold",
    "生成模型",
    "主题",
    "图片编号",
    "场景评分类型",
    "校准范围",
    "训练样本数",
    "center_saliency_ratio",
    "旧center_focus",
    "交叉验证校准值",
    "旧attention",
    "候选attention",
    "attention变化",
    "旧total",
    "候选total",
    "total变化",
    "源JSON",
]

with DETAIL_CSV.open(
    "w",
    newline="",
    encoding="utf-8-sig",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=detail_headers,
    )
    writer.writeheader()
    writer.writerows(detail_rows)


def write_group_csv(path, group_key):
    grouped_rows = defaultdict(list)

    for row in detail_rows:
        grouped_rows[
            row[group_key]
        ].append(row)

    output_rows = []

    for group, group_data in sorted(
        grouped_rows.items(),
        key=lambda item: str(item[0]),
    ):
        attention_stats = describe([
            row["attention变化"]
            for row in group_data
        ])

        total_stats = describe([
            row["total变化"]
            for row in group_data
        ])

        output_rows.append({
            group_key: group,
            "数量": len(group_data),
            "attention平均变化":
                attention_stats["mean"],
            "attention中位变化":
                attention_stats["median"],
            "attention最小变化":
                attention_stats["min"],
            "attention最大变化":
                attention_stats["max"],
            "total平均变化":
                total_stats["mean"],
            "total中位变化":
                total_stats["median"],
            "total最小变化":
                total_stats["min"],
            "total最大变化":
                total_stats["max"],
        })

    headers = [
        group_key,
        "数量",
        "attention平均变化",
        "attention中位变化",
        "attention最小变化",
        "attention最大变化",
        "total平均变化",
        "total中位变化",
        "total最小变化",
        "total最大变化",
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=headers,
        )
        writer.writeheader()
        writer.writerows(output_rows)


write_group_csv(FOLD_CSV, "fold")
write_group_csv(SCENE_CSV, "场景评分类型")


attention_stats = describe([
    row["attention变化"]
    for row in detail_rows
])

total_stats = describe([
    row["total变化"]
    for row in detail_rows
])

summary = {
    "version":
        "transalnet_v1.4.2-attention-candidate-cv",
    "method":
        "5-fold scene-stratified out-of-fold validation",
    "sample_count":
        len(detail_rows),
    "fold_count":
        FOLD_COUNT,
    "attention_delta":
        attention_stats,
    "total_delta":
        total_stats,
    "large_change_counts": {
        "attention_abs_ge_5": sum(
            abs(row["attention变化"]) >= 5
            for row in detail_rows
        ),
        "attention_abs_ge_10": sum(
            abs(row["attention变化"]) >= 10
            for row in detail_rows
        ),
        "total_abs_ge_2": sum(
            abs(row["total变化"]) >= 2
            for row in detail_rows
        ),
    },
}

SUMMARY_JSON.write_text(
    json.dumps(
        summary,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

print("========== 5折交叉验证完成 ==========")
print("样本数：", len(detail_rows))
print("attention变化：", attention_stats)
print("total变化：", total_stats)

print(
    "attention绝对变化>=5：",
    summary["large_change_counts"][
        "attention_abs_ge_5"
    ],
)

print(
    "attention绝对变化>=10：",
    summary["large_change_counts"][
        "attention_abs_ge_10"
    ],
)

print(
    "total绝对变化>=2：",
    summary["large_change_counts"][
        "total_abs_ge_2"
    ],
)

print("详细结果：", DETAIL_CSV)
print("折统计：", FOLD_CSV)
print("场景统计：", SCENE_CSV)
print("汇总结果：", SUMMARY_JSON)
