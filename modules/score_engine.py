"""
Scene-aware rule-based score engine V3.0.

第二阶段修改目标：
1. 信任度和记忆度不再按自然语言字符串数量计分。
2. 使用 evidence_validator.py 输出的 trust_categories 和
   memory_categories，每个证据类别最多计算一次。
3. 根据证据来源设置可靠度：
   OCR直接事实 > Qwen语义判断 > 视觉规则 > 规则推断。
4. 保持 calculate_scores(evidence) 对外接口不变。
5. 当前权重仍是“理论启发的工程初值”，不是最终心理学标定结果。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Mapping


SUPPORTED_SCENES = {
    "电商商品广告",
    "品牌广告",
    "活动宣传海报",
    "短视频封面",
    "教育校园宣传",
    "公益宣传",
    "旅游宣传",
    "其他",
}

SCENE_ALIASES = {
    "电商主图": "电商商品广告",
    "商品主图": "电商商品广告",
    "电商广告": "电商商品广告",
    "产品广告": "电商商品广告",
    "广告海报": "品牌广告",
    "品牌海报": "品牌广告",
    "品牌宣传": "品牌广告",
    "活动海报": "活动宣传海报",
    "直播活动": "活动宣传海报",
    "校园海报": "教育校园宣传",
    "校园活动": "教育校园宣传",
    "教育培训": "教育校园宣传",
    "公益海报": "公益宣传",
    "公益广告": "公益宣传",
    "旅游海报": "旅游宣传",
    "旅游宣传海报": "旅游宣传",
    "旅游广告": "旅游宣传",
    "度假宣传": "旅游宣传",
    "酒店推广": "旅游宣传",
    "度假村推广": "旅游宣传",
}

SOURCE_WEIGHTS = {
    "ocr": 1.00,
    "qwen": 0.88,
    "visual": 0.82,
    "sam": 0.82,
    "rule_inference": 0.62,
}


def _clip_score(value: float) -> int:
    return int(round(max(0.0, min(100.0, float(value)))))


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _nonempty(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return any(_nonempty(item) for item in value)
    if isinstance(value, dict):
        return any(_nonempty(item) for item in value.values())
    return bool(value)


def _normalize_scene_type(scene_type: Any) -> str:
    text = str(scene_type or "").strip()

    if text in SUPPORTED_SCENES:
        return text

    if text in SCENE_ALIASES:
        return SCENE_ALIASES[text]

    for alias, standard in SCENE_ALIASES.items():
        if alias in text:
            return standard

    return "其他"


def _visual(
    evidence: Dict[str, Any],
    key: str,
    default: float = 0.0,
) -> float:
    value = _safe_dict(evidence.get("visual_features")).get(key, default)
    try:
        return float(value if value is not None else default)
    except (TypeError, ValueError):
        return float(default)


def _ratio(count: int, target: int) -> float:
    if target <= 0:
        return 0.0
    return max(0.0, min(1.0, count / target))


def _balance_score(
    value: float,
    target: float,
    tolerance: float,
) -> float:
    if tolerance <= 0:
        return 0.0
    return max(
        0.0,
        1.0 - abs(value - target) / tolerance,
    )


def _quality_score(evidence: Dict[str, Any]) -> float:
    flags = _safe_dict(
        _safe_dict(evidence.get("visual_features")).get("quality_flags")
    )
    problems = sum(
        1
        for key in ("too_dark", "too_bright", "too_cluttered")
        if bool(flags.get(key))
    )
    return max(0.0, 1.0 - 0.25 * problems)


def _semantic_unique_count(values: Any) -> int:
    seen = set()
    count = 0

    for value in _safe_list(values):
        normalized = "".join(
            char.lower()
            for char in str(value)
            if char.isalnum()
        )
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        count += 1

    return count


def _risk_safety(evidence: Dict[str, Any]) -> float:
    risk_count = _semantic_unique_count(evidence.get("risk_points"))
    return max(0.0, 1.0 - min(1.0, risk_count / 4.0))


def _emotion_strength(evidence: Dict[str, Any]) -> float:
    emotion = _safe_dict(evidence.get("emotion_style"))
    label = str(emotion.get("main_emotion", "") or "").strip()
    evidence_count = _semantic_unique_count(emotion.get("evidence"))

    if not label and evidence_count == 0:
        return 0.0

    weak_labels = {"其他", "未知", "无", "理性", "中性"}
    base = 0.35 if label in weak_labels else 0.68
    return min(1.0, base + 0.10 * min(evidence_count, 3))


def _source_strength(sources: Any, inferred: bool = False) -> float:
    clean_sources = [
        str(source).strip()
        for source in _safe_list(sources)
        if str(source).strip()
    ]

    if not clean_sources:
        return 0.62 if inferred else 0.0

    strength = max(
        SOURCE_WEIGHTS.get(source, 0.70)
        for source in clean_sources
    )

    if inferred:
        strength = min(strength, SOURCE_WEIGHTS["rule_inference"])

    return max(0.0, min(1.0, strength))


def _category_strength(category: Any) -> float:
    if isinstance(category, bool):
        return 1.0 if category else 0.0

    if not isinstance(category, dict):
        return 0.0

    if not bool(category.get("present")):
        return 0.0

    return _source_strength(
        category.get("sources"),
        inferred=bool(category.get("inferred")),
    )


def _category_map(
    evidence: Dict[str, Any],
    key: str,
) -> Dict[str, Dict[str, Any]]:
    value = evidence.get(key)
    return value if isinstance(value, dict) else {}


def _category(
    categories: Dict[str, Dict[str, Any]],
    key: str,
) -> float:
    return _category_strength(categories.get(key))


def _coverage(
    categories: Dict[str, Dict[str, Any]],
    keys: Iterable[str],
) -> float:
    key_list = list(keys)
    if not key_list:
        return 0.0
    return sum(
        _category(categories, key)
        for key in key_list
    ) / len(key_list)


def _fallback_trust_categories(
    evidence: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """
    兼容旧版validated_evidence。
    新版validator存在时不会走到这里。
    """
    brand = _safe_dict(evidence.get("brand_info"))
    event = _safe_dict(evidence.get("event_info"))
    price = _safe_dict(evidence.get("price_info"))

    def category(present: bool) -> Dict[str, Any]:
        return {
            "present": bool(present),
            "sources": ["qwen"] if present else [],
            "inferred": False,
            "evidence": [],
        }

    return {
        "brand_identity": category(brand.get("has_brand")),
        "official_channel": category(False),
        "event_time": category(bool(event.get("event_time"))),
        "event_location": category(bool(event.get("event_location"))),
        "organizer_identity": category(bool(event.get("organizer"))),
        "authority_endorsement": category(False),
        "transparent_price": category(price.get("has_price")),
        "concrete_information": category(
            _nonempty(evidence.get("selling_points"))
        ),
        "realistic_evidence": category(False),
    }


def _fallback_memory_categories(
    evidence: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    brand = _safe_dict(evidence.get("brand_info"))
    event = _safe_dict(evidence.get("event_info"))

    def category(present: bool) -> Dict[str, Any]:
        return {
            "present": bool(present),
            "sources": ["qwen"] if present else [],
            "inferred": False,
            "evidence": [],
        }

    return {
        "brand": category(brand.get("has_brand")),
        "event_or_slogan": category(bool(event.get("event_name"))),
        "numeric_symbol": category(False),
        "distinctive_subject": category(
            _nonempty(evidence.get("main_subject"))
        ),
        "visual_symbol": category(False),
        "color_style": category(False),
    }


def _extract_features(evidence: Dict[str, Any]) -> Dict[str, Any]:
    price = _safe_dict(evidence.get("price_info"))
    promo = _safe_dict(evidence.get("promotion_info"))
    campaign = _safe_dict(evidence.get("campaign_info"))
    cta = _safe_dict(evidence.get("cta_info"))
    brand = _safe_dict(evidence.get("brand_info"))
    event = _safe_dict(evidence.get("event_info"))

    trust_categories = _category_map(
        evidence,
        "trust_categories",
    ) or _fallback_trust_categories(evidence)

    memory_categories = _category_map(
        evidence,
        "memory_categories",
    ) or _fallback_memory_categories(evidence)

    risk_count = _semantic_unique_count(
        evidence.get("risk_points")
    )
    attention_count = _semantic_unique_count(
        evidence.get("attention_elements")
    )
    selling_count = _semantic_unique_count(
        evidence.get("selling_points")
    )

    return {
        "scene_type": _normalize_scene_type(
            evidence.get("scene_type")
        ),
        "has_subject": float(
            _nonempty(evidence.get("main_subject"))
        ),
        "has_target_audience": float(
            _nonempty(evidence.get("target_audience"))
        ),
        "selling_count": selling_count,
        "has_price": float(bool(price.get("has_price"))),
        "has_promotion": float(bool(promo.get("has_promotion"))),
        "has_campaign": float(bool(campaign.get("has_campaign"))),
        "has_cta": float(bool(cta.get("has_cta"))),
        "has_event": float(bool(event.get("has_event"))),
        "has_event_name": float(
            bool(str(event.get("event_name", "") or "").strip())
        ),
        "has_event_time": float(
            bool(str(event.get("event_time", "") or "").strip())
        ),
        "has_event_location": float(
            bool(str(event.get("event_location", "") or "").strip())
        ),
        "has_organizer": float(
            bool(str(event.get("organizer", "") or "").strip())
        ),
        "brand_identity": _category(
            trust_categories,
            "brand_identity",
        ),
        "official_channel": _category(
            trust_categories,
            "official_channel",
        ),
        "event_time_trust": _category(
            trust_categories,
            "event_time",
        ),
        "event_location_trust": _category(
            trust_categories,
            "event_location",
        ),
        "organizer_identity": _category(
            trust_categories,
            "organizer_identity",
        ),
        "authority_endorsement": _category(
            trust_categories,
            "authority_endorsement",
        ),
        "transparent_price": _category(
            trust_categories,
            "transparent_price",
        ),
        "concrete_information": _category(
            trust_categories,
            "concrete_information",
        ),
        "realistic_evidence": _category(
            trust_categories,
            "realistic_evidence",
        ),
        "memory_brand": _category(
            memory_categories,
            "brand",
        ),
        "memory_event_or_slogan": _category(
            memory_categories,
            "event_or_slogan",
        ),
        "memory_numeric_symbol": _category(
            memory_categories,
            "numeric_symbol",
        ),
        "memory_distinctive_subject": _category(
            memory_categories,
            "distinctive_subject",
        ),
        "memory_visual_symbol": _category(
            memory_categories,
            "visual_symbol",
        ),
        "memory_color_style": _category(
            memory_categories,
            "color_style",
        ),
        "attention_count": attention_count,
        "risk_count": risk_count,
        "brightness": _visual(evidence, "brightness", 0.5),
        "contrast": _visual(evidence, "contrast", 0.5),
        "saturation": _visual(evidence, "saturation", 0.5),
        "center_focus": _visual(evidence, "center_focus", 0.5),
        "layout_complexity": _visual(
            evidence,
            "layout_complexity",
            0.5,
        ),
        "edge_density": _visual(evidence, "edge_density", 0.1),
        "quality_score": _quality_score(evidence),
        "risk_safety": _risk_safety(evidence),
        "emotion_strength": _emotion_strength(evidence),
        "trust_categories": trust_categories,
        "memory_categories": memory_categories,
    }


def _common_arousal(f: Dict[str, Any]) -> float:
    return (
        28 * f["saturation"]
        + 24 * f["contrast"]
        + 15 * f["center_focus"]
        + 20 * f["emotion_strength"]
        + 13 * _ratio(f["attention_count"], 3)
    )


def _common_attention(
    f: Dict[str, Any],
    ideal_complexity: float,
) -> float:
    return (
        27 * f["contrast"]
        + 22 * f["center_focus"]
        + 21 * _ratio(f["attention_count"], 3)
        + 15 * f["saturation"]
        + 10 * _balance_score(
            f["layout_complexity"],
            ideal_complexity,
            0.55,
        )
        + 5 * f["quality_score"]
    )


def _memory_score(
    f: Dict[str, Any],
    weights: Mapping[str, float],
    ideal_complexity: float,
) -> float:
    return (
        weights.get("brand", 0) * f["memory_brand"]
        + weights.get("event_or_slogan", 0)
        * f["memory_event_or_slogan"]
        + weights.get("numeric_symbol", 0)
        * f["memory_numeric_symbol"]
        + weights.get("distinctive_subject", 0)
        * f["memory_distinctive_subject"]
        + weights.get("visual_symbol", 0)
        * f["memory_visual_symbol"]
        + weights.get("color_style", 0)
        * f["memory_color_style"]
        + weights.get("emotion", 0) * f["emotion_strength"]
        + weights.get("center_focus", 0) * f["center_focus"]
        + weights.get("layout", 0)
        * _balance_score(
            f["layout_complexity"],
            ideal_complexity,
            0.55,
        )
    )


def _score_ecommerce(
    evidence: Dict[str, Any],
    f: Dict[str, Any],
) -> Dict[str, Any]:
    persuasion = (
        14 * f["has_subject"]
        + 22 * _ratio(f["selling_count"], 3)
        + 13 * f["has_price"]
        + 13 * f["has_promotion"]
        + 5 * f["has_campaign"]
        + 14 * f["has_cta"]
        + 10 * f["brand_identity"]
        + 9 * f["center_focus"]
    )

    arousal = _common_arousal(f)

    trust = (
        19 * f["brand_identity"]
        + 10 * f["official_channel"]
        + 14 * f["transparent_price"]
        + 15 * f["concrete_information"]
        + 12 * f["realistic_evidence"]
        + 10 * f["authority_endorsement"]
        + 10 * f["quality_score"]
        + 10 * f["risk_safety"]
    )

    attention = _common_attention(f, ideal_complexity=0.55)

    memory = _memory_score(
        f,
        {
            "brand": 22,
            "event_or_slogan": 12,
            "numeric_symbol": 8,
            "distinctive_subject": 20,
            "visual_symbol": 15,
            "color_style": 8,
            "emotion": 5,
            "center_focus": 5,
            "layout": 5,
        },
        ideal_complexity=0.52,
    )

    return _build_result(
        scene_profile="电商商品广告",
        evidence=evidence,
        f=f,
        raw_scores={
            "persuasion": persuasion,
            "arousal": arousal,
            "trust": trust,
            "attention": attention,
            "memory": memory,
        },
        profile_subrules={
            "persuasion_basis": {
                "has_subject": f["has_subject"],
                "selling_points_count": f["selling_count"],
                "has_price": f["has_price"],
                "has_explicit_promotion": f["has_promotion"],
                "has_campaign_node": f["has_campaign"],
                "has_cta": f["has_cta"],
            },
        },
    )


def _score_brand_ad(
    evidence: Dict[str, Any],
    f: Dict[str, Any],
) -> Dict[str, Any]:
    persuasion = (
        18 * f["has_subject"]
        + 18 * _ratio(f["selling_count"], 3)
        + 21 * f["brand_identity"]
        + 13 * f["has_target_audience"]
        + 16 * f["emotion_strength"]
        + 14 * f["center_focus"]
    )

    arousal = _common_arousal(f)

    trust = (
        25 * f["brand_identity"]
        + 14 * f["official_channel"]
        + 12 * f["authority_endorsement"]
        + 13 * f["concrete_information"]
        + 11 * f["realistic_evidence"]
        + 13 * f["quality_score"]
        + 12 * f["risk_safety"]
    )

    attention = _common_attention(f, ideal_complexity=0.55)

    memory = _memory_score(
        f,
        {
            "brand": 28,
            "event_or_slogan": 16,
            "numeric_symbol": 5,
            "distinctive_subject": 16,
            "visual_symbol": 14,
            "color_style": 8,
            "emotion": 6,
            "center_focus": 4,
            "layout": 3,
        },
        ideal_complexity=0.52,
    )

    return _build_result(
        scene_profile="品牌广告",
        evidence=evidence,
        f=f,
        raw_scores={
            "persuasion": persuasion,
            "arousal": arousal,
            "trust": trust,
            "attention": attention,
            "memory": memory,
        },
        profile_subrules={},
    )


def _score_event_poster(
    evidence: Dict[str, Any],
    f: Dict[str, Any],
) -> Dict[str, Any]:
    event_identity = max(
        f["has_event"],
        f["has_event_name"],
    )
    organizer_or_brand = max(
        f["organizer_identity"],
        f["brand_identity"],
    )

    persuasion = (
        18 * event_identity
        + 10 * f["has_event_time"]
        + 6 * f["has_event_location"]
        + 15 * f["has_cta"]
        + 10 * organizer_or_brand
        + 12 * _ratio(f["selling_count"], 3)
        + 12 * _ratio(f["attention_count"], 3)
        + 7 * f["center_focus"]
        + 5 * f["quality_score"]
        + 5 * f["risk_safety"]
    )

    arousal = (
        29 * f["saturation"]
        + 24 * f["contrast"]
        + 15 * f["center_focus"]
        + 19 * f["emotion_strength"]
        + 13 * _ratio(f["attention_count"], 3)
    )

    trust = (
        18 * f["brand_identity"]
        + 12 * f["official_channel"]
        + 12 * f["event_time_trust"]
        + 8 * f["event_location_trust"]
        + 10 * f["organizer_identity"]
        + 12 * f["authority_endorsement"]
        + 10 * f["concrete_information"]
        + 10 * f["quality_score"]
        + 8 * f["risk_safety"]
    )

    attention = _common_attention(f, ideal_complexity=0.68)

    memory = _memory_score(
        f,
        {
            "brand": 18,
            "event_or_slogan": 22,
            "numeric_symbol": 16,
            "distinctive_subject": 10,
            "visual_symbol": 12,
            "color_style": 8,
            "emotion": 6,
            "center_focus": 4,
            "layout": 4,
        },
        ideal_complexity=0.68,
    )

    return _build_result(
        scene_profile="活动宣传海报",
        evidence=evidence,
        f=f,
        raw_scores={
            "persuasion": persuasion,
            "arousal": arousal,
            "trust": trust,
            "attention": attention,
            "memory": memory,
        },
        profile_subrules={
            "persuasion_basis": {
                "has_event": f["has_event"],
                "has_event_name": f["has_event_name"],
                "has_event_time": f["has_event_time"],
                "has_event_location": f["has_event_location"],
                "has_cta": f["has_cta"],
                "brand_or_organizer_strength": organizer_or_brand,
            },
        },
    )


def _score_video_cover(
    evidence: Dict[str, Any],
    f: Dict[str, Any],
) -> Dict[str, Any]:
    persuasion = (
        20 * f["has_subject"]
        + 16 * f["has_target_audience"]
        + 17 * _ratio(f["selling_count"], 2)
        + 14 * f["has_cta"]
        + 17 * f["emotion_strength"]
        + 16 * f["center_focus"]
    )

    arousal = (
        30 * f["saturation"]
        + 26 * f["contrast"]
        + 18 * f["emotion_strength"]
        + 14 * f["center_focus"]
        + 12 * _ratio(f["attention_count"], 3)
    )

    trust = (
        16 * f["brand_identity"]
        + 11 * f["official_channel"]
        + 14 * f["concrete_information"]
        + 12 * f["realistic_evidence"]
        + 13 * f["authority_endorsement"]
        + 17 * f["quality_score"]
        + 17 * f["risk_safety"]
    )

    attention = (
        31 * f["contrast"]
        + 24 * f["center_focus"]
        + 22 * _ratio(f["attention_count"], 3)
        + 15 * f["saturation"]
        + 8 * _balance_score(
            f["layout_complexity"],
            0.65,
            0.50,
        )
    )

    memory = _memory_score(
        f,
        {
            "brand": 14,
            "event_or_slogan": 22,
            "numeric_symbol": 8,
            "distinctive_subject": 22,
            "visual_symbol": 14,
            "color_style": 7,
            "emotion": 7,
            "center_focus": 4,
            "layout": 2,
        },
        ideal_complexity=0.62,
    )

    return _build_result(
        scene_profile="短视频封面",
        evidence=evidence,
        f=f,
        raw_scores={
            "persuasion": persuasion,
            "arousal": arousal,
            "trust": trust,
            "attention": attention,
            "memory": memory,
        },
        profile_subrules={},
    )


def _score_education(
    evidence: Dict[str, Any],
    f: Dict[str, Any],
) -> Dict[str, Any]:
    responsible_entity = max(
        f["brand_identity"],
        f["organizer_identity"],
    )

    persuasion = (
        18 * f["has_subject"]
        + 20 * _ratio(f["selling_count"], 3)
        + 13 * f["has_target_audience"]
        + 14 * f["has_cta"]
        + 14 * responsible_entity
        + 8 * f["has_event_time"]
        + 7 * f["has_event_location"]
        + 6 * f["center_focus"]
    )

    arousal = (
        22 * f["saturation"]
        + 22 * f["contrast"]
        + 15 * f["center_focus"]
        + 18 * f["emotion_strength"]
        + 13 * _ratio(f["attention_count"], 3)
        + 10 * f["quality_score"]
    )

    trust = (
        18 * f["brand_identity"]
        + 12 * f["official_channel"]
        + 12 * f["organizer_identity"]
        + 10 * f["event_time_trust"]
        + 8 * f["event_location_trust"]
        + 13 * f["authority_endorsement"]
        + 10 * f["concrete_information"]
        + 9 * f["quality_score"]
        + 8 * f["risk_safety"]
    )

    attention = _common_attention(f, ideal_complexity=0.55)

    memory = _memory_score(
        f,
        {
            "brand": 16,
            "event_or_slogan": 20,
            "numeric_symbol": 6,
            "distinctive_subject": 20,
            "visual_symbol": 12,
            "color_style": 7,
            "emotion": 6,
            "center_focus": 7,
            "layout": 6,
        },
        ideal_complexity=0.55,
    )

    return _build_result(
        scene_profile="教育校园宣传",
        evidence=evidence,
        f=f,
        raw_scores={
            "persuasion": persuasion,
            "arousal": arousal,
            "trust": trust,
            "attention": attention,
            "memory": memory,
        },
        profile_subrules={},
    )


def _score_public_welfare(
    evidence: Dict[str, Any],
    f: Dict[str, Any],
) -> Dict[str, Any]:
    responsible_entity = max(
        f["brand_identity"],
        f["organizer_identity"],
    )

    persuasion = (
        21 * f["has_subject"]
        + 20 * f["emotion_strength"]
        + 18 * f["has_cta"]
        + 16 * _ratio(f["selling_count"], 3)
        + 14 * responsible_entity
        + 11 * f["center_focus"]
    )

    arousal = (
        25 * f["saturation"]
        + 23 * f["contrast"]
        + 22 * f["emotion_strength"]
        + 15 * f["center_focus"]
        + 15 * _ratio(f["attention_count"], 3)
    )

    trust = (
        20 * f["brand_identity"]
        + 10 * f["official_channel"]
        + 14 * f["organizer_identity"]
        + 16 * f["authority_endorsement"]
        + 12 * f["concrete_information"]
        + 10 * f["realistic_evidence"]
        + 9 * f["quality_score"]
        + 9 * f["risk_safety"]
    )

    attention = _common_attention(f, ideal_complexity=0.52)

    memory = _memory_score(
        f,
        {
            "brand": 13,
            "event_or_slogan": 21,
            "numeric_symbol": 4,
            "distinctive_subject": 21,
            "visual_symbol": 14,
            "color_style": 8,
            "emotion": 10,
            "center_focus": 5,
            "layout": 4,
        },
        ideal_complexity=0.52,
    )

    return _build_result(
        scene_profile="公益宣传",
        evidence=evidence,
        f=f,
        raw_scores={
            "persuasion": persuasion,
            "arousal": arousal,
            "trust": trust,
            "attention": attention,
            "memory": memory,
        },
        profile_subrules={},
    )



def _score_travel(
    evidence: Dict[str, Any],
    f: Dict[str, Any],
) -> Dict[str, Any]:
    """旅游/酒店/度假目的地宣传的工程初始评分配置。"""
    persuasion = (
        20 * f["has_subject"]
        + 12 * f["has_target_audience"]
        + 14 * _ratio(f["selling_count"], 3)
        + 20 * f["emotion_strength"]
        + 12 * f["center_focus"]
        + 10 * f["realistic_evidence"]
        + 12 * f["quality_score"]
    )

    arousal = (
        21 * f["saturation"]
        + 20 * f["contrast"]
        + 22 * f["emotion_strength"]
        + 13 * f["center_focus"]
        + 12 * _ratio(f["attention_count"], 3)
        + 12 * f["quality_score"]
    )

    trust = (
        8 * f["brand_identity"]
        + 7 * f["official_channel"]
        + 18 * f["realistic_evidence"]
        + 10 * f["concrete_information"]
        + 25 * f["quality_score"]
        + 22 * f["risk_safety"]
        + 10 * f["authority_endorsement"]
    )

    attention = _common_attention(f, ideal_complexity=0.48)

    memory = _memory_score(
        f,
        {
            "brand": 8,
            "event_or_slogan": 8,
            "numeric_symbol": 4,
            "distinctive_subject": 25,
            "visual_symbol": 18,
            "color_style": 12,
            "emotion": 12,
            "center_focus": 7,
            "layout": 6,
        },
        ideal_complexity=0.48,
    )

    return _build_result(
        scene_profile="旅游宣传",
        evidence=evidence,
        f=f,
        raw_scores={
            "persuasion": persuasion,
            "arousal": arousal,
            "trust": trust,
            "attention": attention,
            "memory": memory,
        },
        profile_subrules={
            "travel_basis": {
                "has_subject": f["has_subject"],
                "emotion_strength": round(f["emotion_strength"], 4),
                "realistic_evidence": f["realistic_evidence"],
                "quality_score": f["quality_score"],
            },
        },
    )

def _score_general(
    evidence: Dict[str, Any],
    f: Dict[str, Any],
) -> Dict[str, Any]:
    persuasion = (
        18 * f["has_subject"]
        + 18 * _ratio(f["selling_count"], 3)
        + 10 * f["has_price"]
        + 8 * f["has_promotion"]
        + 4 * f["has_campaign"]
        + 10 * f["has_cta"]
        + 14 * f["brand_identity"]
        + 18 * f["center_focus"]
    )

    arousal = _common_arousal(f)

    trust = (
        20 * f["brand_identity"]
        + 11 * f["official_channel"]
        + 10 * f["authority_endorsement"]
        + 13 * f["concrete_information"]
        + 11 * f["realistic_evidence"]
        + 10 * f["transparent_price"]
        + 13 * f["quality_score"]
        + 12 * f["risk_safety"]
    )

    attention = _common_attention(f, ideal_complexity=0.56)

    memory = _memory_score(
        f,
        {
            "brand": 19,
            "event_or_slogan": 17,
            "numeric_symbol": 7,
            "distinctive_subject": 20,
            "visual_symbol": 14,
            "color_style": 7,
            "emotion": 6,
            "center_focus": 5,
            "layout": 5,
        },
        ideal_complexity=0.56,
    )

    return _build_result(
        scene_profile="通用评分",
        evidence=evidence,
        f=f,
        raw_scores={
            "persuasion": persuasion,
            "arousal": arousal,
            "trust": trust,
            "attention": attention,
            "memory": memory,
        },
        profile_subrules={},
    )


def _build_result(
    scene_profile: str,
    evidence: Dict[str, Any],
    f: Dict[str, Any],
    raw_scores: Dict[str, float],
    profile_subrules: Dict[str, Any],
) -> Dict[str, Any]:
    scores = {
        key: _clip_score(value)
        for key, value in raw_scores.items()
    }
    scores["total"] = round(
        sum(
            scores[key]
            for key in (
                "persuasion",
                "arousal",
                "trust",
                "attention",
                "memory",
            )
        ) / 5,
        2,
    )

    subrules = {
        "scene_type": f["scene_type"],
        "scene_subtype": evidence.get("scene_subtype", ""),
        "source_weights": SOURCE_WEIGHTS,
        "trust_category_strengths": {
            key: round(_category_strength(value), 4)
            for key, value in f["trust_categories"].items()
        },
        "memory_category_strengths": {
            key: round(_category_strength(value), 4)
            for key, value in f["memory_categories"].items()
        },
        "arousal_basis": {
            "saturation": f["saturation"],
            "contrast": f["contrast"],
            "center_focus": f["center_focus"],
            "emotion_strength": round(
                f["emotion_strength"],
                4,
            ),
            "attention_elements_count": f["attention_count"],
        },
        "attention_basis": {
            "contrast": f["contrast"],
            "center_focus": f["center_focus"],
            "saturation": f["saturation"],
            "layout_complexity": f["layout_complexity"],
            "edge_density": f["edge_density"],
            "attention_elements_count": f["attention_count"],
            "quality_score": f["quality_score"],
        },
        "risk_basis": {
            "risk_count": f["risk_count"],
            "risk_safety": f["risk_safety"],
        },
    }
    subrules.update(profile_subrules)

    return {
        "scores": scores,
        "scene_scoring_profile": scene_profile,
        "scoring_version": "scene_category_source_aware_v3.1",
        "subrule_evidence": subrules,
    }


def calculate_scores(
    evidence: Dict[str, Any],
) -> Dict[str, Any]:
    """
    对外统一入口。main.py无需修改。
    """
    evidence = evidence or {}
    features = _extract_features(evidence)
    scene_type = features["scene_type"]

    scorers: Dict[
        str,
        Callable[
            [Dict[str, Any], Dict[str, Any]],
            Dict[str, Any],
        ],
    ] = {
        "电商商品广告": _score_ecommerce,
        "品牌广告": _score_brand_ad,
        "活动宣传海报": _score_event_poster,
        "短视频封面": _score_video_cover,
        "教育校园宣传": _score_education,
        "公益宣传": _score_public_welfare,
        "旅游宣传": _score_travel,
        "其他": _score_general,
    }

    scorer = scorers.get(scene_type, _score_general)
    return scorer(evidence, features)