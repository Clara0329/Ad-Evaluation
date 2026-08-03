#!/usr/bin/env python3

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import modules.score_engine as score_engine


ROOT = Path("outputs/v141_saliency_288")
OUT_DIR = Path("evaluation_saliency_v141")

DETAIL_CSV = OUT_DIR / "center_focus_score_ab_288.csv"
GROUP_CSV = OUT_DIR / "center_focus_score_ab_by_group.csv"
SUMMARY_JSON = OUT_DIR / "center_focus_score_ab_summary.json"

DIMENSIONS = [
    "persuasion",
    "arousal",
    "trust",
    "attention",
    "memory",
    "total",
]


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def get_model_topic(image_path):
    parts = Path(image_path).parts

    try:
        index = parts.index("AIGC_images")
        model = parts[index + 1]
        topic = parts[index + 2]
        return model, topic
    except (ValueError, IndexError):
        return "", ""


def score_difference(new_scores, old_scores, key):
    new_value = safe_float(new_scores.get(key))
    old_value = safe_float(old_scores.get(key))

    if new_value is None or old_value is None:
        return None

    return round(new_value - old_value, 2)


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


OUT_DIR.mkdir(parents=True, exist_ok=True)

json_files = sorted(ROOT.rglob("*_result.json"))

if not json_files:
    raise SystemExit(
        f"没有找到结果JSON：{ROOT}"
    )

# 只在本脚本运行期间临时覆盖_score_engine._visual。
# 不修改正式评分代码和原始JSON。
original_visual = score_engine._visual

override_state = {
    "center_focus": None,
}


def patched_visual(evidence, key, default=0.5):
    if (
        key == "center_focus"
        and override_state["center_focus"] is not None
    ):
        return float(override_state["center_focus"])

    return original_visual(
        evidence,
        key,
        default,
    )


score_engine._visual = patched_visual

rows = []
failed = []
baseline_mismatch = []
trust_changed = []

try:
    for json_path in json_files:
        try:
            data = json.loads(
                json_path.read_text(encoding="utf-8")
            )

            evidence = data.get("validated_evidence")
            saliency = data.get("saliency_analysis", {})
            candidate_features = saliency.get(
                "candidate_features",
                {},
            )

            candidate_value = safe_float(
                candidate_features.get(
                    "center_focus_candidate"
                )
            )

            legacy_value = safe_float(
                candidate_features.get(
                    "legacy_saliency_center"
                )
            )

            if not isinstance(evidence, dict):
                raise ValueError(
                    "缺少validated_evidence"
                )

            if candidate_value is None:
                raise ValueError(
                    "缺少center_focus_candidate"
                )

            stored_scores = (
                data.get("score_result", {})
                .get("scores", {})
            )

            # A组：当前评分引擎按旧center_focus重算
            override_state["center_focus"] = None

            baseline_result = (
                score_engine.calculate_scores(evidence)
            )
            baseline_scores = baseline_result["scores"]

            # B组：只替换center_focus，其余完全不变
            override_state["center_focus"] = candidate_value

            candidate_result = (
                score_engine.calculate_scores(evidence)
            )
            candidate_scores = candidate_result["scores"]

            image_path = data.get("image_path", "")
            model, topic = get_model_topic(image_path)

            stored_matches_baseline = all(
                safe_float(stored_scores.get(key))
                == safe_float(baseline_scores.get(key))
                for key in DIMENSIONS
            )

            if not stored_matches_baseline:
                baseline_mismatch.append(
                    str(json_path)
                )

            trust_delta = score_difference(
                candidate_scores,
                baseline_scores,
                "trust",
            )

            if trust_delta not in (0, 0.0):
                trust_changed.append(
                    str(json_path)
                )

            row = {
                "生成模型": model,
                "主题": topic,
                "图片编号": Path(image_path).stem,
                "图片路径": image_path,
                "场景评分类型":
                    baseline_result.get(
                        "scene_scoring_profile",
                        "",
                    ),
                "旧center_focus": legacy_value,
                "新center_focus_candidate":
                    candidate_value,
                "当前JSON分数与基线重算一致":
                    stored_matches_baseline,
                "结果JSON": str(json_path),
            }

            for key in DIMENSIONS:
                row[f"原JSON_{key}"] = (
                    stored_scores.get(key)
                )
                row[f"基线重算_{key}"] = (
                    baseline_scores.get(key)
                )
                row[f"候选重算_{key}"] = (
                    candidate_scores.get(key)
                )
                row[f"变化_{key}"] = (
                    score_difference(
                        candidate_scores,
                        baseline_scores,
                        key,
                    )
                )

            rows.append(row)

        except Exception as exc:
            failed.append({
                "文件": str(json_path),
                "错误": str(exc),
            })

finally:
    score_engine._visual = original_visual
    override_state["center_focus"] = None


detail_headers = [
    "生成模型",
    "主题",
    "图片编号",
    "图片路径",
    "场景评分类型",
    "旧center_focus",
    "新center_focus_candidate",
    "当前JSON分数与基线重算一致",
]

for dimension in DIMENSIONS:
    detail_headers.extend([
        f"原JSON_{dimension}",
        f"基线重算_{dimension}",
        f"候选重算_{dimension}",
        f"变化_{dimension}",
    ])

detail_headers.append("结果JSON")

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
    writer.writerows(rows)


# 按模型和场景分别统计
groups = defaultdict(list)

for row in rows:
    groups[
        (
            row["生成模型"],
            row["场景评分类型"],
        )
    ].append(row)

group_headers = [
    "生成模型",
    "场景评分类型",
    "数量",
]

for dimension in DIMENSIONS:
    group_headers.extend([
        f"{dimension}_平均变化",
        f"{dimension}_中位变化",
        f"{dimension}_最小变化",
        f"{dimension}_最大变化",
    ])

group_rows = []

for (model, scene), group_data in sorted(
    groups.items()
):
    result = {
        "生成模型": model,
        "场景评分类型": scene,
        "数量": len(group_data),
    }

    for dimension in DIMENSIONS:
        stats = describe([
            item[f"变化_{dimension}"]
            for item in group_data
        ])

        result[f"{dimension}_平均变化"] = (
            stats["mean"]
        )
        result[f"{dimension}_中位变化"] = (
            stats["median"]
        )
        result[f"{dimension}_最小变化"] = (
            stats["min"]
        )
        result[f"{dimension}_最大变化"] = (
            stats["max"]
        )

    group_rows.append(result)

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


overall_changes = {}

for dimension in DIMENSIONS:
    overall_changes[dimension] = describe([
        row[f"变化_{dimension}"]
        for row in rows
    ])

summary = {
    "json_total": len(json_files),
    "success_count": len(rows),
    "failed_count": len(failed),
    "baseline_match_count":
        len(rows) - len(baseline_mismatch),
    "baseline_mismatch_count":
        len(baseline_mismatch),
    "trust_changed_count":
        len(trust_changed),
    "overall_score_changes":
        overall_changes,
    "large_change_counts": {
        "attention_abs_ge_10": sum(
            abs(row["变化_attention"]) >= 10
            for row in rows
            if row["变化_attention"] is not None
        ),
        "total_abs_ge_5": sum(
            abs(row["变化_total"]) >= 5
            for row in rows
            if row["变化_total"] is not None
        ),
    },
    "failed_files": failed,
    "baseline_mismatch_examples":
        baseline_mismatch[:20],
    "trust_changed_examples":
        trust_changed[:20],
}

SUMMARY_JSON.write_text(
    json.dumps(
        summary,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

print("========== A/B重算完成 ==========")
print("JSON总数：", len(json_files))
print("成功数量：", len(rows))
print("失败数量：", len(failed))
print(
    "原JSON与当前基线重算不一致：",
    len(baseline_mismatch),
)
print(
    "信任度发生变化：",
    len(trust_changed),
)

print("\n========== 总体分数变化 ==========")

for dimension in DIMENSIONS:
    stats = overall_changes[dimension]

    print(
        f"{dimension:12s}",
        f"平均={stats['mean']}",
        f"中位={stats['median']}",
        f"最小={stats['min']}",
        f"最大={stats['max']}",
    )

print("\n注意力绝对变化≥10：",
      summary["large_change_counts"][
          "attention_abs_ge_10"
      ])

print("总分绝对变化≥5：",
      summary["large_change_counts"][
          "total_abs_ge_5"
      ])

print("\n详细结果：", DETAIL_CSV)
print("分组结果：", GROUP_CSV)
print("汇总结果：", SUMMARY_JSON)
