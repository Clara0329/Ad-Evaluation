"""Scene classifier V1.3.7: ontology-aware and folder-independent.

It fuses OCR semantics, Qwen visual semantics and hard textual facts. Specific
formats/topics are resolved before generic brand/ecommerce routes. The module
never reads the image path or category folder.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from modules.text_semantics import analyze_text_semantics, clean_text


SCENE_TYPES = {
    "电商商品广告",
    "品牌广告",
    "活动宣传海报",
    "短视频封面",
    "教育校园宣传",
    "公益宣传",
    "旅游宣传",
    "科技创新宣传",
    "其他",
}

SCENE_ALIASES = {
    "电商主图": "电商商品广告",
    "商品主图": "电商商品广告",
    "电商广告": "电商商品广告",
    "产品广告": "电商商品广告",
    "商品广告": "电商商品广告",
    "广告海报": "品牌广告",
    "品牌海报": "品牌广告",
    "品牌宣传": "品牌广告",
    "产品主视觉": "品牌广告",
    "活动海报": "活动宣传海报",
    "直播活动": "活动宣传海报",
    "校园活动": "活动宣传海报",
    "校园比赛": "活动宣传海报",
    "校园海报": "教育校园宣传",
    "教育培训": "教育校园宣传",
    "公益海报": "公益宣传",
    "公益广告": "公益宣传",
    "健康宣传": "公益宣传",
    "旅游海报": "旅游宣传",
    "旅游宣传海报": "旅游宣传",
    "旅游广告": "旅游宣传",
    "度假宣传": "旅游宣传",
    "酒店推广": "旅游宣传",
    "度假村推广": "旅游宣传",
    "科研项目宣传海报": "科技创新宣传",
    "科研宣传": "科技创新宣传",
    "科技宣传": "科技创新宣传",
    "科技创新宣传海报": "科技创新宣传",
}


def _normalize_scene(scene: Any) -> str:
    text = clean_text(scene)
    if text in SCENE_TYPES:
        return text
    if text in SCENE_ALIASES:
        return SCENE_ALIASES[text]
    for alias, standard in SCENE_ALIASES.items():
        if alias in text:
            return standard
    return "其他"


def _qwen_semantic_text(qwen_result: Dict[str, Any]) -> str:
    values: List[str] = []
    for key in (
        "main_subject", "product_type", "scene_subtype", "scene_evidence",
        "visible_evidence", "selling_points", "attention_elements",
        "memory_points", "semantic_inference", "_raw_model_output",
    ):
        value = qwen_result.get(key)
        if isinstance(value, list):
            values.extend(clean_text(item) for item in value)
        elif value:
            values.append(clean_text(value))
    return " ".join(value for value in values if value)


def _has_product_semantics(qwen_result: Dict[str, Any]) -> bool:
    text = _qwen_semantic_text(qwen_result).lower()
    product_words = (
        "鞋", "运动鞋", "车", "汽车", "手机", "耳机", "手表", "咖啡", "饮料",
        "食品", "护肤", "化妆", "香水", "服装", "设备", "包装", "杯", "瓶", "罐",
        "shoe", "sneaker", "car", "vehicle", "phone", "smartphone", "coffee", "drink",
        "skincare", "cosmetic", "perfume", "apparel", "product", "device", "bottle",
        "ai系统", "智能系统", "智能设备", "ai system", "intelligence",
    )
    return any(word in text for word in product_words)


def _contains_any(text: str, values: Tuple[str, ...]) -> bool:
    lower = text.lower()
    return any(value.lower() in lower for value in values)


def refine_scene_by_ocr(
    qwen_scene: str,
    ocr_result: Dict[str, Any],
    qwen_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    qwen_result = qwen_result or {}
    normalized_qwen = _normalize_scene(qwen_scene)
    qwen_brand = clean_text(
        (qwen_result.get("brand_info") or {}).get("brand_text", "")
        if isinstance(qwen_result.get("brand_info"), dict) else ""
    )
    qwen_text_analysis = qwen_result.get("text_analysis") if isinstance(qwen_result.get("text_analysis"), dict) else {}
    text_analysis = analyze_text_semantics(
        ocr_result,
        qwen_brand_candidate=qwen_brand,
        qwen_suspicious_text=qwen_text_analysis.get("suspicious_text", []),
    )

    scores: Dict[str, float] = {
        scene: float(value)
        for scene, value in (text_analysis.get("scene_keyword_scores", {}) or {}).items()
    }
    for scene in SCENE_TYPES:
        scores.setdefault(scene, 0.0)

    trace: List[str] = []
    if normalized_qwen != "其他":
        scores[normalized_qwen] += 2.0
        trace.append(f"Qwen候选={normalized_qwen}")

    qtext = _qwen_semantic_text(qwen_result).lower()
    joined = " ".join(ocr_result.get("all_text", []) or []).lower()
    combined = f"{joined} {qtext}"

    # Qwen visual-semantic bonuses.
    visual_groups: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
        ("旅游宣传", ("海岛", "海滩", "城市风光", "城堡", "酒店", "度假", "island", "beach", "resort", "travel", "landmark")),
        ("公益宣传", ("公益", "健康宣传", "地球", "关爱", "planet", "health awareness", "protect nature", "our planet")),
        ("教育校园宣传", ("教育", "培训", "课程", "学习", "education", "course", "learning")),
        ("短视频封面", ("短视频", "视频封面", "教程", "video thumbnail", "tutorial")),
        ("活动宣传海报", ("活动", "比赛", "竞赛", "发布会", "event", "festival", "competition")),
        ("科技创新宣传", ("科研", "研究项目", "科技创新", "research project", "scientific research", "patent")),
    )
    for scene, markers in visual_groups:
        hits = sum(1 for marker in markers if marker in combined)
        if hits:
            scores[scene] += min(2.4, 0.7 * hits)

    has_price = bool(text_analysis.get("price_text"))
    has_promo = bool(text_analysis.get("promotion_words"))
    has_cta = bool(text_analysis.get("cta_text"))
    has_event_facts = bool(
        text_analysis.get("event_dates")
        or text_analysis.get("event_times")
        or text_analysis.get("event_locations")
    )
    cta_values = [str(value).upper() for value in (text_analysis.get("cta_text") or [])]
    contextual_only_cta = bool(cta_values) and all(
        value in {
            "EXPERIENCE THE FUTURE", "PROTECT NATURE", "REDUCE & REUSE",
            "BUILD A BETTER TOMORROW",
        }
        for value in cta_values
    )

    event_markers = (
        "hackathon", "challenge", "competition", "prizes", "register now",
        "date&venue", "date & venue", "teams of", "top team", "room301",
        "event details", "student union", "main quad", "比赛", "竞赛", "报名",
    )
    short_video_markers = (
        "must watch", "mustwatch", "trend alert", "trending", "click now",
        "short video", "video thumbnail", "教程", "短视频", "必看",
    )
    public_markers = (
        "health awareness", "prevention", "well-being", "stay healthy",
        "healthier community", "get vaccinated", "wash hands", "our planet",
        "protect nature", "our planet", "公益", "健康宣传",
    )
    technology_markers = (
        "research innovation", "researchinnovation", "scientific", "breakthrough",
        "peer-reviewed", "patented", "project name", "research team",
        "publications", "citations", "next-gen battery", "科研", "研究项目", "科技创新",
    )
    tourism_markers = (
        "travel", "destination", "landmark", "heritage", "budapest", "city charm",
        "explore.discover", "explore&", "vacation", "island", "beach", "旅游", "旅行", "度假",
    )
    education_markers = (
        "course", "curriculum", "exam prep", "pass rate", "enroll today",
        "enroll", "learn", "learning", "skills", "learning path", "students",
        "教育", "培训", "课程", "学习", "考试",
    )

    event_hits = sum(marker in combined for marker in event_markers)
    short_hits = sum(marker in combined for marker in short_video_markers)
    public_hits = sum(marker in combined for marker in public_markers)
    tech_hits = sum(marker in combined for marker in technology_markers)
    tourism_hits = sum(marker in combined for marker in tourism_markers)
    education_hits = sum(marker in combined for marker in education_markers)

    # A Qwen ecommerce proposal without price/promotion/action evidence is more
    # consistently a brand poster. Contextual slogans such as EXPERIENCE THE
    # FUTURE are not treated as purchase conversion evidence.
    if normalized_qwen == "电商商品广告" and not has_price and not has_promo and (not has_cta or contextual_only_cta):
        scores["电商商品广告"] = min(scores.get("电商商品广告", 0.0), 0.9)
        scores["品牌广告"] += 0.8
        trace.append("电商候选缺少价格、促销或直接转化CTA，降为品牌展示倾向")

    # A generic Qwen event proposal is rejected without date/location/event nouns.
    if normalized_qwen == "活动宣传海报" and not has_event_facts and event_hits == 0:
        scores["活动宣传海报"] = min(scores.get("活动宣传海报", 0.0), 0.8)
        trace.append("Qwen活动候选缺少日期、地点或活动名词，已降权")

    # Conversion facts are generic evidence and never outrank a strong specific route.
    if has_price:
        scores["电商商品广告"] += 1.8
    if has_promo:
        scores["电商商品广告"] += 1.4
    if has_cta:
        scores["电商商品广告"] += 0.5
    product_semantics = _has_product_semantics(qwen_result)
    if product_semantics:
        scores["品牌广告"] += 1.7
        if has_price or has_promo:
            scores["电商商品广告"] += 0.8
        # Product sustainability/health claims are selling points, not enough
        # evidence to relabel a concrete product poster as public welfare.
        if public_hits < 2:
            scores["公益宣传"] = min(scores.get("公益宣传", 0.0), 1.0)

    forced = ""
    # Order reflects format/topic specificity, not folder labels.
    if (has_event_facts and event_hits >= 1) or event_hits >= 3:
        forced = "活动宣传海报"
        trace.append("检测到日期/地点/报名/竞赛等活动证据")
    elif short_hits >= 1 and not has_event_facts:
        forced = "短视频封面"
        trace.append("检测到MUST WATCH/TRENDING等视频封面语义")
    elif tech_hits >= 2 and not (has_price or has_promo):
        forced = "科技创新宣传"
        trace.append("检测到科研、专利、论文或创新项目证据")
    elif public_hits >= 2 and not (has_price or has_promo) and not product_semantics:
        forced = "公益宣传"
        trace.append("检测到健康/环保倡议类公益语义")
    elif tourism_hits >= 2 and not has_event_facts:
        forced = "旅游宣传"
        trace.append("检测到目的地、探索、地标等旅游语义")
    elif education_hits >= 2 and not has_event_facts:
        forced = "教育校园宣传"
        trace.append("检测到课程、学习或考试培训语义")

    if forced:
        scores[forced] += 4.0
        scene_type = forced
    else:
        specific_order = [
            "活动宣传海报", "短视频封面", "教育校园宣传", "公益宣传",
            "旅游宣传", "科技创新宣传",
        ]
        generic_best = max(scores["品牌广告"], scores["电商商品广告"])
        specific_best = max(specific_order, key=lambda scene: scores.get(scene, 0.0))
        if scores.get(specific_best, 0.0) >= 2.4 and scores[specific_best] >= generic_best - 0.5:
            scene_type = specific_best
        else:
            scene_type = max(SCENE_TYPES, key=lambda scene: scores.get(scene, 0.0))

    if scores.get(scene_type, 0.0) < 1.2:
        scene_type = normalized_qwen if normalized_qwen != "其他" else "其他"

    ordered_scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
    values = list(ordered_scores.values())[:2]
    margin = values[0] - values[1] if len(values) >= 2 else (values[0] if values else 0.0)
    confidence = max(0.0, min(1.0, 0.45 + 0.08 * margin + 0.035 * ordered_scores.get(scene_type, 0.0)))

    reasons = list(trace)
    raw_score = float((text_analysis.get("scene_keyword_scores", {}) or {}).get(scene_type, 0.0) or 0.0)
    if raw_score:
        reasons.append(f"OCR语义关键词分数={raw_score:.1f}")
    if scene_type == "品牌广告" and _has_product_semantics(qwen_result):
        reasons.append("明确产品主体且无更强特定场景证据")
    if scene_type == "电商商品广告":
        facts = [name for name, present in (("价格", has_price), ("促销", has_promo), ("CTA", has_cta)) if present]
        if facts:
            reasons.append("检测到转化信息：" + "、".join(facts))
    if not reasons:
        reasons.append("多源证据不足，采用保守场景判断")

    return {
        "scene_type": scene_type,
        "scene_reasons": reasons,
        "scene_scores": ordered_scores,
        "scene_confidence": round(confidence, 3),
        "scene_decision_trace": trace,
        "text_analysis": text_analysis,
    }
