"""Dual-subject resolver for advertisement understanding (V1.3.6).

The pipeline stores two distinct concepts:
- visual_subject: literal central person/object/symbol visible in the image;
- advertised_subject: product, service, event or destination being promoted.

The advertised subject remains the scoring subject. The visual subject is kept
for provenance and interpretability. Rules are deliberately broad and do not
read category folder names.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _flatten_strings(value: Any) -> List[str]:
    result: List[str] = []
    if value is None:
        return result
    if isinstance(value, str):
        text = re.sub(r"\s+", " ", value).strip()
        if text:
            result.append(text)
        return result
    if isinstance(value, (list, tuple, set)):
        for item in value:
            result.extend(_flatten_strings(item))
        return result
    if isinstance(value, dict):
        for item in value.values():
            result.extend(_flatten_strings(item))
    return result


def _contains_any(text: str, tokens: Iterable[str]) -> bool:
    lower = str(text or "").lower()
    return any(str(token).lower() in lower for token in tokens)


def _detect_faces(image_path: Optional[str]) -> Dict[str, Any]:
    """Best-effort deterministic face presence check using OpenCV Haar cascade."""
    if not image_path or not Path(image_path).exists():
        return {"available": False, "face_count": 0, "reason": "image_missing"}
    try:
        import cv2

        image = cv2.imread(str(image_path))
        if image is None:
            return {"available": False, "face_count": 0, "reason": "imread_failed"}
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        max_side = max(gray.shape[:2])
        scale = min(1.0, 1200.0 / max(max_side, 1))
        if scale < 1.0:
            gray = cv2.resize(gray, None, fx=scale, fy=scale)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
        if cascade.empty():
            return {"available": False, "face_count": 0, "reason": "cascade_unavailable"}
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
        )
        return {
            "available": True,
            "face_count": int(len(faces)),
            "reason": "opencv_haar",
        }
    except Exception as exc:
        return {
            "available": False,
            "face_count": 0,
            "reason": f"{type(exc).__name__}: {exc}",
        }


def resolve_subjects(
    *,
    qwen_result: Dict[str, Any],
    advertised_subject: str,
    product_type: str,
    scene_type: str,
    image_path: Optional[str] = None,
) -> Dict[str, Any]:
    raw_visual = str(qwen_result.get("main_subject", "") or "").strip()
    qwen_product = str(qwen_result.get("product_type", "") or "").strip()
    visual_text = " | ".join(
        _flatten_strings(
            [
                raw_visual,
                qwen_product,
                qwen_result.get("visible_evidence"),
                qwen_result.get("attention_elements"),
                qwen_result.get("memory_points"),
                qwen_result.get("scene_evidence"),
                qwen_result.get("_raw_model_output"),
            ]
        )
    )
    face_info = _detect_faces(image_path)
    face_count = int(face_info.get("face_count", 0) or 0)

    resolved = raw_visual or advertised_subject or product_type
    reasons: List[str] = []
    source = "qwen_literal_subject" if raw_visual else "advertised_subject_fallback"
    confidence = 0.78 if raw_visual else 0.45

    # Container/object correction for coffee packaging.
    if _contains_any(raw_visual, ("咖啡杯", "杯")) and _contains_any(
        f"{qwen_product} {visual_text}", ("玻璃水瓶", "玻璃瓶", "瓶身", "黑色盖子", "提手")
    ):
        resolved = "咖啡瓶"
        reasons.append("Qwen主体为杯，但可见材质、盖子和瓶身证据更符合咖啡瓶")
        source = "visual_rule_resolution"
        confidence = 0.9

    # Normalize visible skincare set rather than generic cosmetics.
    elif _contains_any(raw_visual, ("化妆品套装", "护肤品套装")) and _contains_any(
        f"{qwen_product} {advertised_subject}", ("护肤", "面霜", "精华", "skincare")
    ):
        resolved = "护肤品套装"
        reasons.append("可见容器组与护肤产品语义一致")
        source = "visual_rule_resolution"
        confidence = 0.91

    # Preserve the person when a laptop/computer is used by a visible human.
    elif raw_visual in {"笔记本电脑", "电脑", "平板电脑"} and (
        face_count > 0 or _contains_any(visual_text, ("男人", "女人", "人物", "商务人士", "学生", "教师", "在工作", "使用电脑"))
    ):
        if _contains_any(raw_visual, ("平板",)):
            resolved = "人物使用平板电脑"
        else:
            resolved = "人物使用笔记本电脑"
        reasons.append("可见设备与人物同时构成核心视觉主体")
        source = "qwen_plus_face_resolution" if face_count > 0 else "visual_rule_resolution"
        confidence = 0.88

    # A short-video cover that was mislabeled as software but visibly contains a presenter.
    elif scene_type == "短视频封面" and _contains_any(
        f"{raw_visual} {qwen_product}", ("视频剪辑软件", "视频剪辑", "视频编辑工具")
    ) and (
        face_count > 0 or _contains_any(visual_text, ("男子", "男人", "人物", "讲解者", "教师", "黑板"))
    ):
        resolved = "人物讲解场景"
        reasons.append("短视频封面中检测到人物，软件/工具属于广告语义而非可见主体")
        source = "qwen_plus_face_resolution" if face_count > 0 else "visual_rule_resolution"
        confidence = 0.87

    # Fitness membership posters: people and equipment are the literal subject.
    elif _contains_any(advertised_subject, ("健身房会员", "健身服务")) and (
        face_count > 0 or _contains_any(visual_text, ("健身者", "健身人群", "人物", "训练", "运动员"))
    ):
        resolved = "健身者及健身器材"
        reasons.append("广告服务主体与画面中的健身人物/器材分离记录")
        source = "qwen_plus_face_resolution" if face_count > 0 else "visual_rule_resolution"
        confidence = 0.9

    # Public-welfare imagery: keep the literal symbol instead of an abstract topic label.
    elif scene_type == "公益宣传" and _contains_any(raw_visual, ("环保主题", "公益主题")):
        if _contains_any(visual_text, ("手捧", "手托", "心形", "地球", "生态图案")):
            resolved = "手托地球的环保意象"
            reasons.append("抽象主题被替换为画面中可直接观察的环保视觉符号")
            source = "visual_rule_resolution"
            confidence = 0.86

    changed = bool(resolved and resolved != raw_visual)
    return {
        "visual_subject": resolved,
        "advertised_subject": advertised_subject,
        "main_subject": advertised_subject,
        "product_type": product_type,
        "raw_qwen_visual_subject": raw_visual,
        "changed": changed,
        "source": source,
        "confidence": round(confidence, 3),
        "reasons": reasons,
        "face_evidence": face_info,
    }
