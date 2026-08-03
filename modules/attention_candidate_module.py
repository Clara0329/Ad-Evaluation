from __future__ import annotations

import copy
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np

from modules.score_engine import calculate_scores


MODULE_VERSION = "transalnet_v1.4.3-attention-parallel"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CALIBRATION_PATH = (
    PROJECT_ROOT
    / "configs"
    / "attention_calibration_v143.json"
)

SCORE_DIMENSIONS = [
    "persuasion",
    "arousal",
    "trust",
    "attention",
    "memory",
]


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@lru_cache(maxsize=4)
def _load_calibration(
    calibration_path: str,
) -> Dict[str, Any]:
    path = Path(calibration_path)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():
        raise FileNotFoundError(
            f"校准文件不存在：{path}"
        )

    data = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(data, dict):
        raise ValueError(
            "校准文件根节点必须是字典。"
        )

    return data


def _apply_mapping(
    value: float,
    mapping: Mapping[str, Any],
) -> float:
    source_knots = mapping.get(
        "source_knots",
        [],
    )

    target_knots = mapping.get(
        "target_knots",
        [],
    )

    if not source_knots:
        raise ValueError(
            "校准映射缺少source_knots。"
        )

    if not target_knots:
        raise ValueError(
            "校准映射缺少target_knots。"
        )

    if len(source_knots) != len(target_knots):
        raise ValueError(
            "source_knots与target_knots长度不一致。"
        )

    if len(source_knots) == 1:
        mapped = float(target_knots[0])
    else:
        mapped = float(
            np.interp(
                float(value),
                np.asarray(
                    source_knots,
                    dtype=float,
                ),
                np.asarray(
                    target_knots,
                    dtype=float,
                ),
            )
        )

    mapped = max(0.0, min(1.0, mapped))

    return round(mapped, 4)


def build_attention_candidate(
    validated_evidence: Dict[str, Any],
    official_score_result: Dict[str, Any],
    saliency_analysis: Dict[str, Any],
    calibration_path: str = str(
        DEFAULT_CALIBRATION_PATH
    ),
) -> Dict[str, Any]:
    """
    建立TranSalNet注意力并行候选结果。

    只替换注意力评分使用的center_focus证据。
    不修改正式score_result。
    """

    if not isinstance(
        validated_evidence,
        dict,
    ):
        raise TypeError(
            "validated_evidence必须是字典。"
        )

    if not isinstance(
        official_score_result,
        dict,
    ):
        raise TypeError(
            "official_score_result必须是字典。"
        )

    if not isinstance(
        saliency_analysis,
        dict,
    ):
        raise TypeError(
            "saliency_analysis必须是字典。"
        )

    if saliency_analysis.get("status") != "ok":
        return {
            "status": "skipped",
            "module_version": MODULE_VERSION,
            "scoring_integration": False,
            "reason": "TranSalNet状态不是ok。",
        }

    metrics = saliency_analysis.get(
        "metrics",
        {},
    )

    center_ratio = _safe_float(
        metrics.get(
            "center_saliency_ratio"
        )
    )

    if center_ratio is None:
        return {
            "status": "skipped",
            "module_version": MODULE_VERSION,
            "scoring_integration": False,
            "reason": (
                "缺少center_saliency_ratio。"
            ),
        }

    official_scores = (
        official_score_result.get(
            "scores",
            {},
        )
    )

    if not isinstance(
        official_scores,
        dict,
    ):
        raise ValueError(
            "正式评分中缺少scores。"
        )

    for key in SCORE_DIMENSIONS:
        if _safe_float(
            official_scores.get(key)
        ) is None:
            raise ValueError(
                f"正式评分缺少有效字段：{key}"
            )

    scene = official_score_result.get(
        "scene_scoring_profile",
        "未知场景",
    )

    calibration = _load_calibration(
        calibration_path
    )

    scene_mappings = calibration.get(
        "scene_mappings",
        {},
    )

    mapping = scene_mappings.get(scene)

    if isinstance(mapping, dict):
        calibration_scope = mapping.get(
            "scope",
            "scene",
        )
    else:
        mapping = calibration.get(
            "global_mapping",
            {},
        )
        calibration_scope = (
            "global_fallback"
        )

    if not isinstance(mapping, dict):
        raise ValueError(
            "未找到有效校准映射。"
        )

    calibrated_center = _apply_mapping(
        center_ratio,
        mapping,
    )

    candidate_evidence = copy.deepcopy(
        validated_evidence
    )

    visual_features = (
        candidate_evidence.get(
            "visual_features"
        )
    )

    if not isinstance(
        visual_features,
        dict,
    ):
        visual_features = {}
        candidate_evidence[
            "visual_features"
        ] = visual_features

    legacy_center = _safe_float(
        visual_features.get(
            "center_focus"
        )
    )

    if legacy_center is None:
        legacy_center = _safe_float(
            visual_features.get(
                "saliency_center"
            )
        )

    candidate_features = (
        saliency_analysis.get(
            "candidate_features",
            {},
        )
    )

    if (
        legacy_center is None
        and isinstance(
            candidate_features,
            dict,
        )
    ):
        legacy_center = _safe_float(
            candidate_features.get(
                "legacy_saliency_center"
            )
        )

    visual_features[
        "center_focus"
    ] = calibrated_center

    recalculated_result = (
        calculate_scores(
            candidate_evidence
        )
    )

    recalculated_scores = (
        recalculated_result.get(
            "scores",
            {},
        )
    )

    candidate_attention = _safe_float(
        recalculated_scores.get(
            "attention"
        )
    )

    official_attention = _safe_float(
        official_scores.get(
            "attention"
        )
    )

    official_total = _safe_float(
        official_scores.get(
            "total"
        )
    )

    if candidate_attention is None:
        raise ValueError(
            "候选attention计算失败。"
        )

    if official_attention is None:
        raise ValueError(
            "正式attention缺失。"
        )

    candidate_scores = {
        key: official_scores.get(key)
        for key in SCORE_DIMENSIONS
    }

    candidate_scores[
        "attention"
    ] = int(round(candidate_attention))

    candidate_total = round(
        sum(
            float(candidate_scores[key])
            for key in SCORE_DIMENSIONS
        )
        / len(SCORE_DIMENSIONS),
        2,
    )

    candidate_scores[
        "total"
    ] = candidate_total

    attention_delta = round(
        candidate_scores["attention"]
        - official_attention,
        2,
    )

    total_delta = None

    if official_total is not None:
        total_delta = round(
            candidate_total
            - official_total,
            2,
        )

    return {
        "status": "ok",
        "module_version": MODULE_VERSION,
        "candidate_scope": (
            "attention_only"
        ),
        "scoring_integration": False,
        "scene_scoring_profile": scene,
        "calibration": {
            "path": str(
                Path(calibration_path)
            ),
            "version": calibration.get(
                "version"
            ),
            "method": calibration.get(
                "method"
            ),
            "scope": calibration_scope,
            "training_sample_size":
                calibration.get(
                    "training_sample_size"
                ),
        },
        "center_focus_evidence": {
            "legacy_center_focus":
                legacy_center,
            "center_saliency_ratio":
                center_ratio,
            "calibrated_center_focus":
                calibrated_center,
        },
        "official_scores":
            official_scores,
        "candidate_scores":
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
        "note": (
            "实验性并行候选结果；"
            "只替换注意力评分中的center_focus证据，"
            "正式score_result未修改。"
        ),
    }
