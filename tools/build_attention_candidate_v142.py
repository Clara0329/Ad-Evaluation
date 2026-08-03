#!/usr/bin/env python3

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

import modules.score_engine as score_engine


SOURCE_ROOT = Path("outputs/v141_saliency_288")
OUTPUT_ROOT = Path("outputs/v142_attention_candidate_288")
EVAL_ROOT = Path("evaluation_saliency_v142")

DETAIL_CSV = EVAL_ROOT / "attention_candidate_v142_288.csv"
GROUP_CSV = EVAL_ROOT / "attention_candidate_v142_by_scene.csv"
SUMMARY_JSON = EVAL_ROOT / "attention_candidate_v142_summary.json"
CALIBRATION_JSON = EVAL_ROOT / "center_focus_scene_calibration_v142.json"

VERSION = "transalnet_v1.4.2-attention-candidate"

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

MIN_SCENE_SAMPLE = 12

SCORE_DIMENSIONS = [
    "persuasion",
    "arousal",
    "trust",
    "attention",
    "memory",
]


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def describe(values):
    clean = [
        float(value)
        for value in values
        if value is not None
    ]

    if not clean:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
        }

    return {
        "count": len(clean),
        "mean": round(statistics.mean(clean), 4),
        "median": round(statistics.median(clean), 4),
        "min": round(min(clean), 4),
        "max": round(max(clean), 4),
    }


def create_quantile_mapping(records):
    raw_values = np.asarray(
        [record["raw_candidate"] for record in records],
        dtype=float,
    )

    legacy_values = np.asarray(
        [record["legacy_center_focus"] for record in records],
        dtype=float,
    )

    x_knots = np.quantile(
        raw_values,
        QUANTILES,
    )

    y_knots = np.quantile(
        legacy_values,
        QUANTILES,
    )

    # 去除重复横坐标，保证np.interp可正常使用
    unique_x = []
    unique_y = []

    for x_value, y_value in zip(x_knots, y_knots):
        x_value = float(x_value)
        y_value = float(y_value)

        if (
            unique_x
            and abs(x_value - unique_x[-1]) < 1e-12
        ):
            unique_y[-1] = max(
                unique_y[-1],
                y_value,
            )
        else:
            unique_x.append(x_value)
            unique_y.append(y_value)

    return {
        "quantiles": QUANTILES,
        "raw_candidate_knots": [
            round(value, 6)
            for value in unique_x
        ],
        "legacy_center_focus_knots": [
            round(value, 6)
            for value in unique_y
        ],
        "sample_size": len(records),
    }


def apply_mapping(value, mapping):
    x_values = mapping["raw_candidate_knots"]
    y_values = mapping["legacy_center_focus_knots"]

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

    mapped = max(0.0, min(1.0, float(mapped)))

    return round(mapped, 4)


def scores_match(left, right):
    for key in SCORE_DIMENSIONS + ["total"]:
        if safe_float(left.get(key)) != safe_float(
            right.get(key)
        ):
            return False

    return True


json_files = sorted(
    SOURCE_ROOT.rglob("*_result.json")
)

if not json_files:
    raise SystemExit(
        f"没有找到结果JSON：{SOURCE_ROOT}"
    )

records = []
failed = []

for json_path in json_files:
    try:
        data = json.loads(
            json_path.read_text(encoding="utf-8")
        )

        saliency = data.get(
            "saliency_analysis",
            {},
        )

        candidate_features = saliency.get(
            "candidate_features",
            {},
        )

        raw_candidate = safe_float(
            candidate_features.get(
                "center_focus_candidate"
            )
        )

        legacy_center = safe_float(
            candidate_features.get(
                "legacy_saliency_center"
            )
        )

        evidence = data.get(
            "validated_evidence"
        )

        score_result = data.get(
            "score_result",
            {},
        )

        old_scores = score_result.get(
            "scores",
            {},
        )

        scene = score_result.get(
            "scene_scoring_profile",
            "未知场景",
        )

        if raw_candidate is None:
            raise ValueError(
                "缺少center_focus_candidate"
            )

        if legacy_center is None:
            raise ValueError(
                "缺少legacy_saliency_center"
            )

        if not isinstance(evidence, dict):
            raise ValueError(
                "缺少validated_evidence"
            )

        records.append({
            "json_path": json_path,
            "data": data,
            "image_path": data.get(
                "image_path",
                "",
            ),
            "scene": scene,
            "raw_candidate": raw_candidate,
            "legacy_center_focus": legacy_center,
            "evidence": evidence,
            "old_scores": old_scores,
        })

    except Exception as exc:
        failed.append({
            "file": str(json_path),
            "error": str(exc),
        })


if failed:
    print("发现无法读取的结果：")

    for item in failed:
        print(item)

    raise SystemExit(
        "存在无效JSON，停止生成候选结果。"
    )


# 建立全局映射
global_mapping = create_quantile_mapping(
    records
)

# 建立场景内映射
records_by_scene = defaultdict(list)

for record in records:
    records_by_scene[
        record["scene"]
    ].append(record)

scene_mappings = {}

for scene, scene_records in records_by_scene.items():
    unique_raw_count = len({
        record["raw_candidate"]
        for record in scene_records
    })

    if (
        len(scene_records) >= MIN_SCENE_SAMPLE
        and unique_raw_count >= 3
    ):
        scene_mappings[scene] = {
            "scope": "scene",
            **create_quantile_mapping(
                scene_records
            ),
        }
    else:
        scene_mappings[scene] = {
            "scope": "global_fallback",
            **global_mapping,
        }


calibration_payload = {
    "version": VERSION,
    "method": "scene_quantile_mapping",
    "description": (
        "将TranSalNet中心显著候选值映射到旧center_focus的"
        "场景内分布，仅用于注意力候选分。"
    ),
    "minimum_scene_sample": MIN_SCENE_SAMPLE,
    "global_mapping": global_mapping,
    "scene_mappings": scene_mappings,
}

EVAL_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

CALIBRATION_JSON.write_text(
    json.dumps(
        calibration_payload,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


# 临时覆盖score_engine中的center_focus读取
original_visual = score_engine._visual

override_state = {
    "center_focus": None,
}


def patched_visual(
    evidence,
    key,
    default=0.5,
):
    if (
        key == "center_focus"
        and override_state["center_focus"]
        is not None
    ):
        return float(
            override_state["center_focus"]
        )

    return original_visual(
        evidence,
        key,
        default,
    )


score_engine._visual = patched_visual

detail_rows = []
baseline_mismatch = []

try:
    for record in records:
        scene = record["scene"]
        mapping = scene_mappings[scene]

        calibrated_center = apply_mapping(
            record["raw_candidate"],
            mapping,
        )

        # 检查当前评分代码与原JSON是否一致
        override_state["center_focus"] = None

        baseline_result = (
            score_engine.calculate_scores(
                record["evidence"]
            )
        )

        baseline_scores = baseline_result[
            "scores"
        ]

        if not scores_match(
            baseline_scores,
            record["old_scores"],
        ):
            baseline_mismatch.append(
                str(record["json_path"])
            )

        # 用校准后的中心特征重算
        override_state[
            "center_focus"
        ] = calibrated_center

        full_candidate_result = (
            score_engine.calculate_scores(
                record["evidence"]
            )
        )

        recalculated_scores = (
            full_candidate_result["scores"]
        )

        # 只采用新的attention
        # 其余四个维度保持正式旧分数
        candidate_scores = {
            key: record["old_scores"][key]
            for key in SCORE_DIMENSIONS
        }

        candidate_scores["attention"] = (
            recalculated_scores["attention"]
        )

        candidate_scores["total"] = round(
            sum(
                candidate_scores[key]
                for key in SCORE_DIMENSIONS
            ) / 5,
            2,
        )

        old_attention = safe_float(
            record["old_scores"].get(
                "attention"
            )
        )

        new_attention = safe_float(
            candidate_scores.get(
                "attention"
            )
        )

        old_total = safe_float(
            record["old_scores"].get(
                "total"
            )
        )

        new_total = safe_float(
            candidate_scores.get(
                "total"
            )
        )

        attention_delta = round(
            new_attention - old_attention,
            2,
        )

        total_delta = round(
            new_total - old_total,
            2,
        )

        relative_path = (
            record["json_path"]
            .relative_to(SOURCE_ROOT)
        )

        output_path = (
            OUTPUT_ROOT
            / relative_path.parent
            / (
                relative_path.stem
                .replace(
                    "_result",
                    "",
                )
                + "_attention_candidate.json"
            )
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        sidecar = {
            "source_result_json": str(
                record["json_path"]
            ),
            "image_path": record[
                "image_path"
            ],
            "module_version": VERSION,
            "scene_scoring_profile": scene,
            "scoring_integration": False,
            "candidate_scope": (
                "attention_only"
            ),
            "center_focus_evidence": {
                "legacy_center_focus":
                    record[
                        "legacy_center_focus"
                    ],
                "raw_center_focus_candidate":
                    record[
                        "raw_candidate"
                    ],
                "calibrated_center_focus":
                    calibrated_center,
                "calibration_scope":
                    mapping["scope"],
                "calibration_method":
                    "scene_quantile_mapping",
            },
            "score_comparison": {
                "official_scores":
                    record["old_scores"],
                "attention_candidate_scores":
                    candidate_scores,
                "delta": {
                    "persuasion": 0,
                    "arousal": 0,
                    "trust": 0,
                    "attention":
                        attention_delta,
                    "memory": 0,
                    "total": total_delta,
                },
            },
            "note": (
                "实验性候选结果。正式score_result未修改。"
            ),
        }

        output_path.write_text(
            json.dumps(
                sidecar,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        path_parts = Path(
            record["image_path"]
        ).parts

        model = ""
        topic = ""

        try:
            image_index = (
                path_parts.index(
                    "AIGC_images"
                )
            )

            model = path_parts[
                image_index + 1
            ]

            topic = path_parts[
                image_index + 2
            ]

        except (
            ValueError,
            IndexError,
        ):
            pass

        detail_rows.append({
            "生成模型": model,
            "主题": topic,
            "图片编号": Path(
                record["image_path"]
            ).stem,
            "场景评分类型": scene,
            "旧center_focus":
                record[
                    "legacy_center_focus"
                ],
            "原始TranSalNet候选值":
                record[
                    "raw_candidate"
                ],
            "场景校准候选值":
                calibrated_center,
            "校准范围":
                mapping["scope"],
            "旧attention":
                old_attention,
            "候选attention":
                new_attention,
            "attention变化":
                attention_delta,
            "旧total":
                old_total,
            "候选total":
                new_total,
            "total变化":
                total_delta,
            "源JSON":
                str(record["json_path"]),
            "候选JSON":
                str(output_path),
        })

finally:
    score_engine._visual = (
        original_visual
    )

    override_state[
        "center_focus"
    ] = None


detail_headers = [
    "生成模型",
    "主题",
    "图片编号",
    "场景评分类型",
    "旧center_focus",
    "原始TranSalNet候选值",
    "场景校准候选值",
    "校准范围",
    "旧attention",
    "候选attention",
    "attention变化",
    "旧total",
    "候选total",
    "total变化",
    "源JSON",
    "候选JSON",
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


group_rows = []

for scene in sorted(
    records_by_scene
):
    scene_rows = [
        row
        for row in detail_rows
        if row[
            "场景评分类型"
        ] == scene
    ]

    attention_stats = describe([
        row["attention变化"]
        for row in scene_rows
    ])

    total_stats = describe([
        row["total变化"]
        for row in scene_rows
    ])

    group_rows.append({
        "场景评分类型": scene,
        "数量": len(scene_rows),
        "校准范围":
            scene_mappings[
                scene
            ]["scope"],
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


group_headers = [
    "场景评分类型",
    "数量",
    "校准范围",
    "attention平均变化",
    "attention中位变化",
    "attention最小变化",
    "attention最大变化",
    "total平均变化",
    "total中位变化",
    "total最小变化",
    "total最大变化",
]

with GROUP_CSV.open(
    "w",
    newline="",
    encoding="utf-8-sig",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=group_headers,
    )

    writer.writeheader()
    writer.writerows(group_rows)


attention_stats = describe([
    row["attention变化"]
    for row in detail_rows
])

total_stats = describe([
    row["total变化"]
    for row in detail_rows
])

summary = {
    "version": VERSION,
    "source_json_count":
        len(json_files),
    "success_count":
        len(detail_rows),
    "baseline_mismatch_count":
        len(baseline_mismatch),
    "candidate_scope":
        "attention_only",
    "official_score_modified":
        False,
    "attention_delta":
        attention_stats,
    "total_delta":
        total_stats,
    "large_change_counts": {
        "attention_abs_ge_5": sum(
            abs(
                row["attention变化"]
            ) >= 5
            for row in detail_rows
        ),
        "attention_abs_ge_10": sum(
            abs(
                row["attention变化"]
            ) >= 10
            for row in detail_rows
        ),
        "total_abs_ge_2": sum(
            abs(
                row["total变化"]
            ) >= 2
            for row in detail_rows
        ),
    },
    "baseline_mismatch_examples":
        baseline_mismatch[:20],
}

SUMMARY_JSON.write_text(
    json.dumps(
        summary,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


print(
    "========== v1.4.2候选完成 =========="
)

print(
    "源JSON数量：",
    len(json_files),
)

print(
    "成功数量：",
    len(detail_rows),
)

print(
    "基线不一致：",
    len(baseline_mismatch),
)

print(
    "\nattention变化：",
    attention_stats,
)

print(
    "total变化：",
    total_stats,
)

print(
    "\nattention绝对变化>=5：",
    summary[
        "large_change_counts"
    ][
        "attention_abs_ge_5"
    ],
)

print(
    "attention绝对变化>=10：",
    summary[
        "large_change_counts"
    ][
        "attention_abs_ge_10"
    ],
)

print(
    "total绝对变化>=2：",
    summary[
        "large_change_counts"
    ][
        "total_abs_ge_2"
    ],
)

print(
    "\n详细结果：",
    DETAIL_CSV,
)

print(
    "场景结果：",
    GROUP_CSV,
)

print(
    "汇总结果：",
    SUMMARY_JSON,
)

print(
    "候选JSON目录：",
    OUTPUT_ROOT,
)
