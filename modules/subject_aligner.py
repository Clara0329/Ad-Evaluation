"""Cross-modal subject alignment for advertisement understanding.

This module reconciles three evidence sources without reading file paths or
folder names:
1. OCR semantic cues (what product/service the poster talks about);
2. Qwen visible-object cues (what is physically visible in the image);
3. the resolved scene type.

The goal is not to guess a specific brand/model. It only normalizes the main
subject to a conservative semantic category when OCR and visual evidence agree
that the raw Qwen subject is a container, prop, person, or other secondary
object.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from modules.text_semantics import analyze_text_semantics, clean_text, normalize_text


@dataclass(frozen=True)
class SubjectProfile:
    key: str
    subject: str
    product_type: str
    scene_hint: str
    keyword_weights: Mapping[str, float]
    visual_hints: Tuple[str, ...] = ()
    minimum_score: float = 2.5


PROFILES: Tuple[SubjectProfile, ...] = (
    SubjectProfile(
        key="footwear",
        subject="运动鞋",
        product_type="运动鞋",
        scene_hint="品牌广告",
        keyword_weights={
            "shoe": 2.2, "sneaker": 2.2, "running": 1.4, "outsole": 1.2,
            "cushion": 1.1, "traction": 1.1, "breathable": 0.8, "pace": 0.7,
            "运动鞋": 2.5, "跑步鞋": 2.5, "鞋底": 1.1,
        },
        visual_hints=("鞋", "运动鞋", "跑步鞋", "鞋底", "鞋面"),
    ),
    SubjectProfile(
        key="skincare",
        subject="护肤品套装",
        product_type="美妆护肤品",
        scene_hint="品牌广告",
        keyword_weights={
            "skincare": 2.4, "beauty": 1.0, "cream": 1.4, "serum": 1.4,
            "elixir": 1.2, "radiance": 0.8, "nourish": 0.9, "botanical": 0.8,
            "护肤": 2.5, "美容": 1.2, "面霜": 1.6, "精华": 1.6,
        },
        visual_hints=("护肤", "化妆品", "瓶", "罐", "套装", "面霜", "精华"),
    ),
    SubjectProfile(
        key="vehicle",
        subject="汽车",
        product_type="汽车",
        scene_hint="品牌广告",
        keyword_weights={
            "vehicle": 2.1, "electric": 1.3, "mobility": 1.2, "drive": 0.8,
            "autonomous": 1.1, "range": 0.8, "suv": 1.7, "car": 2.1,
            "汽车": 2.5, "电动汽车": 2.5, "新能源": 1.3, "续航": 0.8,
        },
        visual_hints=("汽车", "轿车", "SUV", "车身", "轮毂"),
    ),
    SubjectProfile(
        key="smartphone",
        subject="智能手机",
        product_type="智能手机",
        scene_hint="品牌广告",
        keyword_weights={
            "smartphone": 2.5, "phone": 2.0, "amoled": 1.4, "camera": 0.8,
            "mah": 1.0, "5g": 1.0, "display": 0.8, "battery": 0.8,
            "手机": 2.5, "智能手机": 2.5, "摄像头": 0.8, "电池": 0.8,
        },
        visual_hints=("手机", "智能手机", "摄像头", "机身"),
    ),
    SubjectProfile(
        key="coffee",
        subject="咖啡饮品",
        product_type="咖啡饮品",
        scene_hint="电商商品广告",
        keyword_weights={
            "coffee": 2.5, "arabica": 1.4, "beans": 0.9, "brewed": 1.0,
            "roasted": 1.0, "fresh brew": 1.1, "咖啡": 2.5, "咖啡豆": 1.1,
        },
        visual_hints=("咖啡", "杯", "瓶", "饮品", "保温杯", "玻璃瓶"),
    ),
    SubjectProfile(
        key="education",
        subject="在线教育培训课程",
        product_type="教育培训服务",
        scene_hint="教育校园宣传",
        keyword_weights={
            "learn": 1.1, "learning": 1.2, "skills": 1.2, "course": 1.5,
            "enroll": 1.4, "learners": 1.0, "job-ready": 1.1, "career": 0.9,
            "certificate": 1.1, "online": 0.8, "教育": 2.2, "培训": 2.2,
            "课程": 1.8, "学习": 1.2, "招生": 1.2,
        },
        visual_hints=("学习", "教育", "课程", "学生", "教师", "平板电脑", "笔记本电脑"),
        minimum_score=3.0,
    ),
    SubjectProfile(
        key="ai_system",
        subject="AI智能系统",
        product_type="科技创新产品",
        scene_hint="品牌广告",
        keyword_weights={
            " ai ": 1.8, "intelligence": 1.5, "assistant": 1.1, "analytics": 1.0,
            "natural language": 1.2, "smart data": 1.0, "automate": 0.9,
            "optimize": 0.8, "人工智能": 2.2, "智能系统": 1.6, "智能助手": 1.5,
        },
        visual_hints=("智能", "AI", "音箱", "设备", "屏幕", "界面", "笔记本电脑"),
        minimum_score=2.8,
    ),
    SubjectProfile(
        key="public_welfare",
        subject="环保公益主题",
        product_type="公益宣传内容",
        scene_hint="公益宣传",
        keyword_weights={
            "planet": 1.5, "earth": 1.5, "responsibility": 1.0,
            "sustainable": 1.2, "protect nature": 1.4, "reduce": 0.7,
            "reuse": 0.7, "环保": 2.2, "地球": 1.5, "可持续": 1.4,
        },
        visual_hints=("地球", "植物", "森林", "环保", "自然"),
        minimum_score=2.5,
    ),
    SubjectProfile(
        key="short_video",
        subject="知识技能教程短视频",
        product_type="短视频内容",
        scene_hint="短视频封面",
        keyword_weights={
            "short video": 2.2, "video": 1.0, "tutorial": 1.5,
            "tips & tricks": 1.4, "beginners": 1.1, "steps": 1.0,
            "in30days": 1.0, "master": 0.8, "skill": 0.9,
            "短视频": 2.2, "教程": 1.5, "技巧": 1.0, "新手": 1.0,
        },
        visual_hints=("视频", "教程", "人物", "相机", "手机", "图标"),
        minimum_score=2.6,
    ),
    SubjectProfile(
        key="fitness_service",
        subject="健身房会员服务",
        product_type="健身服务",
        scene_hint="电商商品广告",
        keyword_weights={
            "fitness": 1.8, "gym": 1.8, "member": 1.2, "membership": 1.3,
            "join now": 1.0, "onboarding": 1.0, "first month free": 1.3,
            "unlimited": 0.8, "健身房": 2.0, "会员": 1.3, "健身服务": 1.8,
        },
        visual_hints=("健身房", "健身人群", "器材", "训练", "健身设备"),
        minimum_score=3.0,
    ),
    SubjectProfile(
        key="tourism",
        subject="热带海岛旅游场景",
        product_type="旅游度假服务",
        scene_hint="旅游宣传",
        keyword_weights={
            "tropical": 1.2, "island": 1.5, "paradise": 1.2, "beach": 1.2,
            "resort": 1.2, "getaway": 1.0, "escape": 0.8, "ocean": 0.8,
            "海岛": 2.0, "旅游": 2.0, "度假": 1.5, "海滩": 1.2,
        },
        visual_hints=("海滩", "海岛", "度假村", "情侣", "沙滩", "海水"),
        minimum_score=2.5,
    ),
    SubjectProfile(
        key="campus_event",
        subject="校园社团活动",
        product_type="校园活动",
        scene_hint="活动宣传海报",
        keyword_weights={
            "club": 1.4, "event details": 1.5, "student union": 1.5,
            "main quad": 1.4, "join our club": 1.8, "club fair": 1.8,
            "社团": 2.0, "校园活动": 2.0, "学生会": 1.4,
        },
        visual_hints=("学生", "社团", "活动", "卡通人物", "帐篷", "气球"),
        minimum_score=2.8,
    ),
)


SECONDARY_OR_GENERIC_SUBJECT_HINTS = (
    "男人", "女人", "学生", "情侣", "人物", "模特", "跑者", "背景人物",
    "笔记本电脑", "平板电脑", "咖啡杯", "水瓶", "玻璃水瓶", "保温杯",
    "健身设备", "健身器材", "器材", "摄像机", "视频制作工具", "视频剪辑软件",
    "智能音箱", "设备", "产品", "主题", "内容",
)

UNVERIFIED_BRAND_OR_MODEL_PATTERNS = (
    re.compile(r"\b(?:iphone|samsung|apple|huawei|xiaomi|oppo|vivo|new balance)\b", re.I),
    re.compile(r"\b[A-Za-z]{2,}\s*\d{1,4}\b"),
)


def _combined_ocr_text(ocr_result: Dict[str, Any]) -> str:
    lines = [clean_text(x) for x in (ocr_result.get("all_text") or []) if clean_text(x)]
    return " ".join(lines)


def _qwen_visual_text(qwen_result: Dict[str, Any], subject: str, product_type: str) -> str:
    values: List[str] = [subject, product_type]
    for key in ("visible_evidence", "selling_points", "attention_elements", "memory_points"):
        value = qwen_result.get(key)
        if isinstance(value, list):
            values.extend(clean_text(x) for x in value)
        elif value:
            values.append(clean_text(value))
    return " ".join(x for x in values if x)


def _keyword_score(text_lower: str, profile: SubjectProfile) -> Tuple[float, List[str]]:
    score = 0.0
    hits: List[str] = []
    padded = f" {text_lower} "
    for keyword, weight in profile.keyword_weights.items():
        key = keyword.lower()
        target = padded if key.startswith(" ") or key.endswith(" ") else text_lower
        if key in target:
            score += float(weight)
            hits.append(keyword.strip())
    return score, hits


def _visual_compatibility(visual_text: str, profile: SubjectProfile) -> float:
    lowered = visual_text.lower()
    hits = sum(1 for hint in profile.visual_hints if hint.lower() in lowered)
    return min(1.5, hits * 0.45)


def _scene_bonus(scene_type: str, profile: SubjectProfile) -> float:
    if scene_type == profile.scene_hint:
        return 1.2
    if profile.key == "short_video" and scene_type == "教育校园宣传":
        return 0.2
    if profile.key == "education" and scene_type == "短视频封面":
        return -0.8
    return 0.0


def _is_secondary_or_generic(subject: str, product_type: str) -> bool:
    combined = f"{subject} {product_type}".strip()
    if not combined:
        return True
    return any(hint in combined for hint in SECONDARY_OR_GENERIC_SUBJECT_HINTS)


def _contains_unverified_brand_or_model(value: str) -> bool:
    return any(pattern.search(value or "") for pattern in UNVERIFIED_BRAND_OR_MODEL_PATTERNS)


def _profile_specific_subject(
    profile: SubjectProfile,
    visual_text: str,
) -> Tuple[str, str]:
    """Use conservative visual refinement without inventing brand/model details."""
    lowered = visual_text.lower()
    if profile.key == "coffee":
        if any(token in lowered for token in ("瓶", "玻璃水瓶", "玻璃瓶")):
            return "瓶装咖啡饮品", "咖啡饮品"
        return "咖啡饮品", "咖啡饮品"
    if profile.key == "ai_system":
        if any(token in lowered for token in ("智能音箱", "音箱", "圆柱", "圆柱形", "智能终端")):
            return "AI智能设备", "科技创新产品"
        return profile.subject, profile.product_type
    return profile.subject, profile.product_type


def align_main_subject(
    *,
    ocr_result: Dict[str, Any],
    qwen_result: Dict[str, Any],
    scene_type: str,
    current_subject: str,
    current_product_type: str,
    text_analysis: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return a conservative cross-modal subject alignment decision."""
    text_analysis = text_analysis or analyze_text_semantics(ocr_result)
    ocr_text = _combined_ocr_text(ocr_result)
    text_lower = ocr_text.lower()
    visual_text = _qwen_visual_text(qwen_result, current_subject, current_product_type)

    candidates: List[Dict[str, Any]] = []
    for profile in PROFILES:
        keyword_score, hits = _keyword_score(text_lower, profile)
        if not hits:
            continue
        total = keyword_score + _visual_compatibility(visual_text, profile) + _scene_bonus(scene_type, profile)
        candidates.append({
            "profile": profile,
            "score": round(total, 3),
            "keyword_score": round(keyword_score, 3),
            "hits": hits,
        })

    candidates.sort(key=lambda item: item["score"], reverse=True)
    if not candidates:
        return {
            "main_subject": current_subject,
            "product_type": current_product_type,
            "source": "qwen" if current_subject else "",
            "changed": False,
            "confidence": 0.0,
            "profile": "",
            "reasons": [],
            "candidates": [],
        }

    best = candidates[0]
    second_score = candidates[1]["score"] if len(candidates) > 1 else 0.0
    profile: SubjectProfile = best["profile"]
    margin = best["score"] - second_score

    scene_consistent = scene_type == profile.scene_hint
    generic_current = _is_secondary_or_generic(current_subject, current_product_type)
    unsupported_specific = _contains_unverified_brand_or_model(
        f"{current_subject} {current_product_type}"
    )

    # For service/experience scenes, OCR semantics should override a visible prop/person.
    semantic_service_profile = profile.key in {
        "education", "short_video", "fitness_service", "tourism",
        "campus_event", "public_welfare", "ai_system",
    }

    threshold = profile.minimum_score
    strong_enough = best["score"] >= threshold
    sufficiently_separated = margin >= 0.8 or best["score"] >= threshold + 1.5
    footwear_specific_cleanup = (
        profile.key == "footwear"
        and current_product_type not in {"", "运动鞋", "跑步鞋"}
    )
    skincare_specific_cleanup = (
        profile.key == "skincare"
        and current_subject in {"化妆品套装", "美容产品套装"}
    )
    should_change = strong_enough and sufficiently_separated and (
        generic_current
        or unsupported_specific
        or semantic_service_profile
        or footwear_specific_cleanup
        or skincare_specific_cleanup
        or not current_subject
    )

    # Do not let education absorb a short-video cover when video semantics are stronger.
    if profile.key == "education" and scene_type == "短视频封面":
        should_change = False

    if should_change:
        aligned_subject, aligned_product = _profile_specific_subject(profile, visual_text)
        confidence = min(0.95, 0.55 + 0.06 * best["score"] + 0.05 * max(margin, 0.0))
        reasons = [
            f"OCR语义命中：{', '.join(best['hits'][:6])}",
            f"场景={scene_type}，跨模态类别={profile.key}",
        ]
        if generic_current:
            reasons.append(f"原主体“{current_subject or current_product_type}”更像人物、容器或陪衬对象")
        if unsupported_specific:
            reasons.append("原主体或产品类型包含未验证品牌/型号")
        return {
            "main_subject": aligned_subject,
            "product_type": aligned_product,
            "source": "cross_modal_alignment",
            "changed": (aligned_subject != current_subject or aligned_product != current_product_type),
            "confidence": round(confidence, 3),
            "profile": profile.key,
            "reasons": reasons,
            "candidates": [
                {"profile": item["profile"].key, "score": item["score"], "hits": item["hits"][:8]}
                for item in candidates[:3]
            ],
        }

    return {
        "main_subject": current_subject,
        "product_type": current_product_type,
        "source": "qwen" if current_subject else "",
        "changed": False,
        "confidence": 0.0,
        "profile": profile.key,
        "reasons": [],
        "candidates": [
            {"profile": item["profile"].key, "score": item["score"], "hits": item["hits"][:8]}
            for item in candidates[:3]
        ],
    }
