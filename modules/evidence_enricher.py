"""Deterministic evidence enrichment V1.3.

This module enriches Qwen evidence with facts already extracted from OCR. It does
not decide brands or scenes and does not invent product functions.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List

from modules.text_semantics import analyze_text_semantics, clean_text, normalize_text


NUMERIC_INFO_PATTERN = re.compile(
    r"(?:[$€£￥¥]\s*\d+(?:\.\d+)?)|(?:\d+(?:\.\d+)?\s*(?:%|mp|mah|gb|tb|hz|km|ml|days?|hours?|mins?|分钟|小时|天))",
    re.I,
)


def _dedupe(items: Iterable[Any]) -> List[str]:
    result: List[str] = []
    seen = set()
    for item in items:
        text = clean_text(item).strip(" ，。；;")
        key = normalize_text(text)
        if text and key and key not in seen:
            seen.add(key)
            result.append(text)
    return result


def _qwen_list(qwen_result: Dict[str, Any], key: str) -> List[str]:
    value = qwen_result.get(key, [])
    return _dedupe(value if isinstance(value, list) else [])


def _ocr_concrete_information(lines: List[str]) -> List[str]:
    return _dedupe(line for line in lines if NUMERIC_INFO_PATTERN.search(line))[:6]


def _attention_from_visual(visual: Dict[str, Any]) -> List[str]:
    elements: List[str] = []
    contrast = float(visual.get("contrast", 0.0) or 0.0)
    saturation = float(visual.get("saturation", 0.0) or 0.0)
    center_focus = float(visual.get("center_focus", 0.0) or 0.0)
    complexity = float(visual.get("layout_complexity", 0.0) or 0.0)
    if center_focus >= 0.78:
        elements.append("主体位于中心区域，视觉焦点集中")
    if contrast >= 0.48:
        elements.append("主体与背景明暗对比明显")
    if saturation >= 0.68:
        elements.append("高饱和色彩形成强视觉刺激")
    if complexity <= 0.32:
        elements.append("构图简洁，视觉竞争较少")
    elif complexity >= 0.82:
        elements.append("信息和视觉元素较密集")
    return elements


def _memory_from_text(text_analysis: Dict[str, Any]) -> List[str]:
    result: List[str] = []
    for cta in text_analysis.get("cta_text", [])[:1]:
        result.append(f"行动号召：{cta}")
    for promotion in text_analysis.get("promotion_words", [])[:1]:
        result.append(f"促销信息：{promotion}")
    for price in text_analysis.get("price_text", [])[:1]:
        result.append(f"价格数字：{price}")
    return result


def _infer_emotion(
    qwen_emotion: Dict[str, Any],
    visual: Dict[str, Any],
    scene_type: str,
) -> Dict[str, Any]:
    main = clean_text(qwen_emotion.get("main_emotion", "")) if isinstance(qwen_emotion, dict) else ""
    evidence = _dedupe(qwen_emotion.get("evidence", []) if isinstance(qwen_emotion, dict) else [])
    if main and main not in {"其他", "理性", "理性专业"} and evidence:
        return {"main_emotion": main, "evidence": evidence[:4]}

    saturation = float(visual.get("saturation", 0.0) or 0.0)
    contrast = float(visual.get("contrast", 0.0) or 0.0)

    if scene_type == "旅游宣传":
        main = "温暖愉悦"
        evidence = ["旅游与度假场景传达放松和向往感"]
    elif scene_type == "公益宣传":
        main = "关怀责任"
        evidence = ["公益倡议语义强调责任、关怀与共同参与"]
    elif scene_type == "短视频封面" and (saturation >= 0.55 or contrast >= 0.55):
        main = "热烈兴奋"
        evidence = ["高对比封面构图强化即时注意和行动感"]
    elif saturation >= 0.67:
        main = "热烈兴奋"
        evidence = [f"整体饱和度较高（{saturation:.2f}）"]
    elif saturation <= 0.25:
        main = "平静治愈"
        evidence = [f"整体饱和度较低（{saturation:.2f}）"]
    else:
        main = main or "理性专业"
        evidence = evidence or [f"整体色彩较克制（饱和度{saturation:.2f}）"]
    if contrast >= 0.48:
        evidence.append(f"明暗对比较明显（{contrast:.2f}）")
    return {"main_emotion": main, "evidence": _dedupe(evidence)[:4]}


def enrich_evidence(
    ocr_result: Dict[str, Any],
    visual_result: Dict[str, Any],
    qwen_result: Dict[str, Any],
) -> Dict[str, Any]:
    lines = [clean_text(x) for x in ocr_result.get("all_text", []) if clean_text(x)]
    qwen_text_analysis = qwen_result.get("text_analysis") if isinstance(qwen_result.get("text_analysis"), dict) else {}
    qwen_brand = ""
    if isinstance(qwen_result.get("brand_info"), dict):
        qwen_brand = clean_text(qwen_result["brand_info"].get("brand_text", ""))
    text_analysis = analyze_text_semantics(
        ocr_result,
        qwen_brand_candidate=qwen_brand,
        qwen_suspicious_text=qwen_text_analysis.get("suspicious_text", []),
    )

    visible_evidence = _qwen_list(qwen_result, "visible_evidence")
    qwen_points = _qwen_list(qwen_result, "selling_points")
    selling_points = _dedupe(visible_evidence + qwen_points + _ocr_concrete_information(lines))[:8]

    trust_signals = _dedupe(_qwen_list(qwen_result, "trust_signals") + _ocr_concrete_information(lines))[:8]
    attention_elements = _dedupe(
        _qwen_list(qwen_result, "attention_elements") + _attention_from_visual(visual_result)
    )[:8]
    memory_points = _dedupe(
        _qwen_list(qwen_result, "memory_points")
        + visible_evidence
        + _memory_from_text(text_analysis)
    )[:8]

    scene_type = clean_text(qwen_result.get("scene_type", "其他"))
    emotion_style = _infer_emotion(
        qwen_result.get("emotion_style", {}),
        visual_result,
        scene_type,
    )

    confidence = qwen_result.get("confidence", {})
    if not isinstance(confidence, dict):
        confidence = {}
    subject_ok = bool(clean_text(qwen_result.get("main_subject", "")))
    fallback = bool(qwen_result.get("fallback"))
    computed = {
        "main_subject": 0.85 if subject_ok else 0.35,
        "ocr_understanding": min(0.95, 0.50 + min(len(lines), 20) * 0.02),
        "emotion_judgment": 0.70 if emotion_style.get("evidence") else 0.45,
        "overall": 0.25 if fallback else min(0.92, 0.55 + (0.10 if subject_ok else 0.0) + min(len(selling_points), 4) * 0.04),
    }
    for key, value in computed.items():
        try:
            existing = float(confidence.get(key, 0.0) or 0.0)
        except (TypeError, ValueError):
            existing = 0.0
        confidence[key] = round(max(existing, value), 3)

    return {
        "selling_points": selling_points,
        "trust_signals": trust_signals,
        "attention_elements": attention_elements,
        "memory_points": memory_points,
        "emotion_style": emotion_style,
        "confidence": confidence,
        "text_analysis": text_analysis,
    }
