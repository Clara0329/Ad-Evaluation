"""
Evidence validator V3.0 — category-deduplicated, source-aware evidence fusion.

第二阶段修改目标：
1. 不再使用 event_name = main_subject 的强制补全。
2. 信任与记忆证据按“证据类别”计数，不按自然语言字符串数量计数。
3. 记录关键字段的证据来源：ocr、qwen、visual、sam、rule_inference。
4. 保持原有 validate_evidence(...) 接口不变，兼容 main.py。
5. 保留原有自然语言证据列表，便于解释和展示。
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from modules.evidence_enricher import enrich_evidence
from modules.text_semantics import analyze_text_semantics, clean_text, is_watermark, normalize_text
from modules.subject_aligner import align_main_subject
from modules.subject_resolver import resolve_subjects
from modules.text_role_classifier import classify_text_roles

try:
    from modules.scene_classifier import refine_scene_by_ocr
except ImportError:
    def refine_scene_by_ocr(
        qwen_scene: str,
        ocr_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        scene = str(qwen_scene or "").strip() or "其他"
        return {
            "scene_type": scene,
            "scene_reasons": [
                "scene_classifier.py 未加载，沿用Qwen场景结果"
            ],
        }


SCENE_NAMES = {
    "电商商品广告",
    "品牌广告",
    "活动宣传海报",
    "短视频封面",
    "教育校园宣传",
    "公益宣传",
    "旅游宣传",
    "科技创新宣传",
    "其他",
    # 兼容旧名称
    "电商主图",
    "广告海报",
    "活动海报",
}

EVENT_TITLE_HINTS = (
    "活动",
    "直播",
    "发布会",
    "购物节",
    "音乐节",
    "电影节",
    "艺术节",
    "文化节",
    "狂欢节",
    "发烧夜",
    "盛典",
    "晚会",
    "大会",
    "论坛",
    "峰会",
    "展览",
    "展会",
    "比赛",
    "竞赛",
    "挑战赛",
    "联赛",
    "招新",
    "讲座",
    "公开课",
    "训练营",
    "体验日",
    "品牌日",
    "周年庆",
)

EVENT_MARKERS = (
    "直播时间",
    "活动时间",
    "演出时间",
    "比赛时间",
    "报名时间",
    "活动地点",
    "演出地点",
    "比赛地点",
    "主办方",
    "承办方",
    "嘉宾",
    "开幕",
    "直播",
    "发布会",
    "音乐节",
    "购物节",
)

EXPLICIT_PROMOTION_HINTS = (
    "满减",
    "立减",
    "折扣",
    "优惠券",
    "到手价",
    "优惠价",
    "特价",
    "秒杀",
    "限时抢购",
    "买一送一",
    "赠品",
    "免单",
    "包邮",
)

CAMPAIGN_HINTS = (
    "购物节",
    "大促",
    "狂欢",
    "周年庆",
    "品牌日",
    "年中",
    "年货",
    "开门红",
    "直播节",
)

CTA_HINTS = (
    "立即购买",
    "马上购买",
    "点击购买",
    "立即抢购",
    "立即参与",
    "立即报名",
    "扫码参与",
    "扫码报名",
    "了解更多",
    "点击查看",
    "点击观看",
    "预约直播",
)

OFFICIAL_CHANNEL_HINTS = (
    "官网",
    "官方",
    "旗舰店",
    "认证账号",
    "官方账号",
)

AUTHORITY_HINTS = (
    "权威",
    "认证",
    "专家",
    "机构",
    "协会",
    "研究院",
    "联合发布",
    "合作伙伴",
    "代言",
    "嘉宾",
)

REALISTIC_EVIDENCE_HINTS = (
    "真实场景",
    "实拍",
    "参数",
    "检测报告",
    "用户评价",
    "销量",
    "口碑",
    "案例",
    "成分",
    "材质",
)

VISUAL_SYMBOL_HINTS = (
    "logo",
    "图标",
    "吉祥物",
    "人物阵容",
    "独特造型",
    "标志性",
    "视觉符号",
    "插画形象",
    "光轨",
    "光带",
    "动感线条",
    "速度线",
    "流线",
    "倒影",
    "剪影",
    "几何线条",
    "悬浮",
    "对称构图",
    "建筑造型",
)

COLOR_STYLE_HINTS = (
    "红",
    "蓝",
    "黄",
    "绿",
    "青",
    "紫",
    "橙",
    "粉",
    "金色",
    "黑金",
    "撞色",
    "渐变",
    "高饱和",
    "低饱和",
    "高对比",
    "冷色",
    "暖色",
    "冷暖",
    "光影",
)

BRAND_STOPWORDS = {
    "立即购买",
    "限时抢购",
    "新品上市",
    "购物节",
    "促销活动",
    "广告海报",
    "活动海报",
    "短视频封面",
    "直播时间",
    "活动时间",
    "演出时间",
    "比赛时间",
}

TEXTUAL_ELEMENT_HINTS = (
    "标题", "文案", "文字", "参数区", "参数", "价格区", "价格",
    "信息区", "按钮", "购买区", "主标题", "副标题",
)

DIRECT_VISUAL_HINTS = (
    "色", "构图", "光", "背景", "质感", "造型", "外观", "轮廓",
    "主体", "居中", "对比", "渐变", "光轨", "动感", "运动感",
    "极简", "简洁", "金属", "透明", "悬浮", "摄影", "场景",
    "纹理", "屏幕", "镜头", "瓶身", "包装", "杯体", "鞋底",
    "车身", "轮毂", "人物", "表情", "姿态", "图标", "logo",
    "高饱和", "低饱和", "明暗", "阴影", "大型", "中央",
)

FUNCTIONAL_CLAIM_HINTS = (
    "舒适", "耐磨", "防滑", "节能", "高效", "性能", "处理器",
    "夜景模式", "保湿", "补水", "续航", "降噪", "音质", "健康",
    "安全", "智能科技", "功效", "效果好", "便携实用",
)


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


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
        preferred_keys = (
            "text",
            "value",
            "content",
            "description",
            "name",
            "evidence",
        )
        for key in preferred_keys:
            if key in value:
                result.extend(_flatten_strings(value[key]))
        return result

    text = str(value).strip()
    if text:
        result.append(text)
    return result


def _ensure_list(value: Any) -> List[str]:
    return _flatten_strings(value)


def _normalize_text(text: Any) -> str:
    value = str(text or "").lower()
    value = re.sub(r"[\s，。、“”‘’：:；;！!？?（）()\[\]【】<>《》\-_/\\|·]+", "", value)
    return value


def _has_meaningful_ocr_text(ocr_result: Dict[str, Any]) -> bool:
    """过滤单字装饰符号，至少存在一个长度>=2的OCR片段才视为有效文字。"""
    return any(
        len(_normalize_text(item)) >= 2
        for item in _ocr_lines(ocr_result)
    )


def _text_supported(value: Any, ocr_result: Dict[str, Any]) -> bool:
    value_norm = _normalize_text(value)
    joined_norm = _normalize_text(_ocr_joined_text(ocr_result))
    return bool(value_norm) and bool(joined_norm) and value_norm in joined_norm


def _is_direct_visual_evidence(value: Any) -> bool:
    text = str(value or "").strip()
    return bool(text) and _contains_any(text, DIRECT_VISUAL_HINTS)


def _is_unverifiable_functional_claim(value: Any) -> bool:
    text = str(value or "").strip()
    return bool(text) and _contains_any(text, FUNCTIONAL_CLAIM_HINTS)


def _normalize_emotion_label(value: Any) -> str:
    text = str(value or "").strip()
    aliases = {
        "热情": "热烈兴奋",
        "兴奋": "热烈兴奋",
        "活力": "热烈兴奋",
        "愉悦": "温暖愉悦",
        "温馨": "温暖愉悦",
        "理性": "理性专业",
        "专业": "理性专业",
        "刺激": "紧张刺激",
        "治愈": "平静治愈",
        "平静": "平静治愈",
        "关怀": "悲伤关怀",
    }
    return aliases.get(text, text or "其他")


def _filter_visual_or_ocr_evidence(
    values: Any,
    ocr_result: Dict[str, Any],
    notes: Optional[List[str]] = None,
    field_name: str = "证据",
    max_items: int = 8,
) -> List[str]:
    """
    只保留OCR直接支持或描述可见视觉事实的证据。
    这一步用于抑制“保湿、节能、高性能”等产品常识推测。
    """
    kept: List[str] = []
    removed: List[str] = []

    for item in _unique_strings(values):
        text_supported = _text_supported(item, ocr_result)
        if _is_unverifiable_functional_claim(item) and not text_supported:
            removed.append(item)
        elif text_supported or _is_direct_visual_evidence(item):
            kept.append(item)
        else:
            removed.append(item)

    if removed and notes is not None:
        notes.append(
            f"{field_name}中删除了缺少OCR或直接视觉依据的推测项："
            + "、".join(removed[:4])
        )

    return kept[:max_items]


def _filter_textual_elements_without_text(
    values: Any,
    ocr_result: Dict[str, Any],
    notes: Optional[List[str]] = None,
    field_name: str = "视觉证据",
    max_items: int = 8,
) -> List[str]:
    items = _unique_strings(values)
    if _has_meaningful_ocr_text(ocr_result):
        return items[:max_items]

    kept = [
        item for item in items
        if not _contains_any(item, TEXTUAL_ELEMENT_HINTS)
    ]
    removed = [item for item in items if item not in kept]
    if removed and notes is not None:
        notes.append(
            f"OCR无有效文字，已删除{field_name}中关于标题/文案/参数区的描述。"
        )
    return kept[:max_items]


def _visual_feature_fallbacks(
    visual_result: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[str], List[str]]:
    """根据可解释的基础视觉特征补充情绪、注意力与记忆证据。"""
    def number(key: str, default: float = 0.0) -> float:
        try:
            return float(visual_result.get(key, default) or default)
        except (TypeError, ValueError):
            return default

    saturation = number("saturation", 0.0)
    contrast = number("contrast", 0.0)
    center_focus = number("center_focus", 0.0)
    complexity = number("layout_complexity", 0.0)

    emotion_evidence: List[str] = []
    if saturation >= 0.65:
        emotion = "热烈兴奋"
        emotion_evidence.append(f"整体饱和度较高（{saturation:.2f}）")
    elif saturation <= 0.25 and complexity <= 0.35:
        emotion = "平静治愈"
        emotion_evidence.append(f"整体饱和度较低（{saturation:.2f}）且构图较简洁")
    else:
        emotion = "理性专业"
        emotion_evidence.append(f"整体色彩与构图较克制（饱和度{saturation:.2f}）")

    if contrast >= 0.45:
        emotion_evidence.append(f"明暗对比较明显（{contrast:.2f}）")

    attention: List[str] = []
    if center_focus >= 0.8:
        attention.append("主体位于中心区域，视觉焦点集中")
    if contrast >= 0.45:
        attention.append("主体与背景明暗对比明显")
    if saturation >= 0.65:
        attention.append("高饱和色彩形成较强视觉刺激")
    if complexity <= 0.35:
        attention.append("构图较简洁，主体竞争较少")

    memory: List[str] = []
    if saturation >= 0.65:
        memory.append("高饱和色彩风格")
    if contrast >= 0.45:
        memory.append("高对比光影风格")

    return (
        {"main_emotion": emotion, "evidence": emotion_evidence[:3]},
        attention[:4],
        memory[:3],
    )



_COLOR_REQUIREMENTS = {
    "红色": {"red", "pink"},
    "橙色": {"orange"},
    "黄色": {"yellow", "orange"},
    "绿色": {"green", "cyan"},
    "青色": {"cyan", "green", "blue"},
    "蓝色": {"blue", "cyan"},
    "紫色": {"purple", "pink"},
    "粉色": {"pink", "red"},
    "金色": {"yellow", "orange"},
}


def _dominant_color_labels(image_path: Optional[str]) -> set[str]:
    """从原图提取主导色相标签，仅用于校验明确颜色词，不参与直接评分。"""
    if not image_path:
        return set()

    try:
        import colorsys
        from collections import defaultdict
        from PIL import Image

        path = str(image_path)
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            rgb.thumbnail((96, 96))
            pixels = list(rgb.getdata())

        weights = defaultdict(float)
        for red, green, blue in pixels:
            r, g, b = red / 255.0, green / 255.0, blue / 255.0
            hue, saturation, value = colorsys.rgb_to_hsv(r, g, b)
            if saturation < 0.24 or value < 0.10:
                continue
            degree = hue * 360.0
            if degree < 15 or degree >= 345:
                label = "red"
            elif degree < 45:
                label = "orange"
            elif degree < 75:
                label = "yellow"
            elif degree < 165:
                label = "green"
            elif degree < 200:
                label = "cyan"
            elif degree < 255:
                label = "blue"
            elif degree < 315:
                label = "purple"
            else:
                label = "pink"
            weights[label] += saturation * max(value, 0.25)

        total = sum(weights.values())
        if total <= 0:
            return set()

        ordered = sorted(weights.items(), key=lambda item: item[1], reverse=True)
        labels = {label for label, weight in ordered[:3]}
        labels.update(
            label for label, weight in ordered
            if weight / total >= 0.08
        )
        return labels
    except Exception:
        return set()


def _required_color_groups(text: str) -> List[set[str]]:
    text = str(text or "")
    groups: List[set[str]] = []
    for phrase, accepted in _COLOR_REQUIREMENTS.items():
        if phrase in text:
            groups.append(accepted)

    # 兼容“橙蓝高对比”“蓝绿渐变”等省略“色”字的组合写法。
    if any(marker in text for marker in ("对比", "撞色", "渐变", "配色")):
        short_map = {
            "红": {"red", "pink"},
            "橙": {"orange"},
            "黄": {"yellow", "orange"},
            "绿": {"green", "cyan"},
            "青": {"cyan", "green", "blue"},
            "蓝": {"blue", "cyan"},
            "紫": {"purple", "pink"},
            "粉": {"pink", "red"},
        }
        for token, accepted in short_map.items():
            if token in text:
                groups.append(accepted)
    return groups


def _filter_unsupported_color_claims(
    items: List[str],
    image_path: Optional[str],
    notes: Optional[List[str]],
    field_name: str,
) -> List[str]:
    labels = _dominant_color_labels(image_path)
    if not labels:
        return items

    kept: List[str] = []
    for item in items:
        requirements = _required_color_groups(item)
        if requirements and any(not (group & labels) for group in requirements):
            if notes is not None:
                notes.append(
                    f"{field_name}中的明确颜色描述与原图主导色不一致，已删除：{item}。"
                )
            continue
        kept.append(item)
    return kept

def _unique_strings(*values: Any, max_items: Optional[int] = None) -> List[str]:
    result: List[str] = []
    seen = set()

    for value in values:
        for item in _ensure_list(value):
            normalized = _normalize_text(item)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            result.append(item)
            if max_items is not None and len(result) >= max_items:
                return result

    return result


def _contains_any(text: str, keywords: Sequence[str]) -> bool:
    return any(keyword.lower() in text.lower() for keyword in keywords)


def _ocr_lines(ocr_result: Dict[str, Any]) -> List[str]:
    return _unique_strings(ocr_result.get("all_text"))


def _ocr_joined_text(ocr_result: Dict[str, Any]) -> str:
    joined = str(ocr_result.get("joined_text", "") or "").strip()
    return joined if joined else " ".join(_ocr_lines(ocr_result))


def _extract_domains(text: str) -> List[str]:
    domains = re.findall(
        r"(?<![\w.-])(?:https?://)?(?:www\.)?([a-z0-9-]+(?:\.[a-z0-9-]+)+)",
        text.lower(),
    )
    return _unique_strings(domains)


def _source_record(
    value: Any,
    sources: Iterable[str],
    inferred: bool = False,
    evidence: Optional[List[str]] = None,
) -> Dict[str, Any]:
    clean_sources = _unique_strings(list(sources))
    return {
        "value": value,
        "sources": clean_sources,
        "inferred": bool(inferred),
        "evidence": _unique_strings(evidence or []),
    }


def _new_category() -> Dict[str, Any]:
    return {
        "present": False,
        "evidence": [],
        "sources": [],
        "inferred": False,
    }


def _add_category(
    categories: Dict[str, Dict[str, Any]],
    category_name: str,
    evidence_text: Any,
    source: str,
    inferred: bool = False,
) -> None:
    category = categories.setdefault(category_name, _new_category())
    category["present"] = True
    category["evidence"] = _unique_strings(
        category.get("evidence"),
        evidence_text,
    )
    category["sources"] = _unique_strings(
        category.get("sources"),
        [source],
    )
    category["inferred"] = bool(category.get("inferred")) or bool(inferred)


def _resolve_scene(
    qwen_result: Dict[str, Any],
    ocr_result: Dict[str, Any],
) -> Tuple[str, List[str], Dict[str, Any]]:
    qwen_scene = str(qwen_result.get("scene_type", "") or "").strip()

    try:
        scene_result = refine_scene_by_ocr(
            qwen_scene=qwen_scene,
            ocr_result=ocr_result,
            qwen_result=qwen_result,
        )
    except TypeError:
        try:
            scene_result = refine_scene_by_ocr(
                qwen_scene=qwen_scene,
                ocr_result=ocr_result,
            )
        except TypeError:
            scene_result = refine_scene_by_ocr(qwen_scene, ocr_result)

    if isinstance(scene_result, dict):
        scene_type = str(
            scene_result.get("scene_type", qwen_scene or "其他") or "其他"
        ).strip()
        reasons = _unique_strings(scene_result.get("scene_reasons"))
    elif isinstance(scene_result, str) and scene_result.strip():
        scene_type = scene_result.strip()
        reasons = []
    else:
        scene_type = qwen_scene or "其他"
        reasons = []

    sources: List[str] = []
    if qwen_scene:
        sources.append("qwen")

    inferred = scene_type != qwen_scene
    if inferred:
        sources.append("rule_inference")

    if any("OCR" in reason for reason in reasons):
        sources.append("ocr")
    if any("产品主体" in reason or "产品主视觉" in reason for reason in reasons):
        sources.append("qwen")
        sources.append("rule_inference")

    provenance = _source_record(
        value=scene_type,
        sources=sources or ["rule_inference"],
        inferred=inferred,
        evidence=reasons,
    )
    return scene_type, reasons, provenance



def _is_domain(text: str) -> bool:
    return bool(
        re.fullmatch(
            r"(?:https?://)?(?:www\.)?[\w-]+(?:\.[\w-]+)+",
            text.strip().lower(),
        )
    )


def _brand_candidate_score(
    candidate: str,
    all_lines: List[str],
    domains: List[str],
) -> float:
    text = candidate.strip()
    normalized = _normalize_text(text)

    if not normalized:
        return -100.0

    # 单个汉字、单个字母或装饰符号极易是OCR误识别，禁止作为品牌。
    if len(normalized) < 2:
        return -100.0

    if text in BRAND_STOPWORDS or _contains_any(text, tuple(BRAND_STOPWORDS)):
        return -20.0

    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        return -20.0

    if _contains_any(text, EVENT_TITLE_HINTS):
        return -5.0

    if _contains_any(text, ("时间", "地点", "日期", "价格", "优惠", "立即")):
        return -10.0

    score = 0.0

    if _is_domain(text):
        score += 4.0

    if re.search(r"[\u4e00-\u9fff]", text):
        score += 2.0

    if 2 <= len(text) <= 12:
        score += 2.0

    if all_lines and text == all_lines[0]:
        score += 1.0

    # 若候选文本与域名前缀相近，增加可信度。
    compact = re.sub(r"[^a-z0-9]", "", text.lower())
    for domain in domains:
        stem = domain.split(".")[0]
        if compact and stem and (compact in stem or stem in compact):
            score += 2.0

    return score


def _select_ocr_brand_candidate(
    ocr_result: Dict[str, Any],
    qwen_candidate: str = "",
    qwen_suspicious_text: Optional[Sequence[str]] = None,
    text_roles: Optional[Dict[str, Any]] = None,
) -> Tuple[str, List[str], str]:
    """Return only a confirmed OCR brand; ambiguous text stays a candidate."""
    role_data = _safe_dict(text_roles)
    candidates = [
        item for item in role_data.get("brand_candidates", [])
        if isinstance(item, dict)
    ]
    confirmed = [item for item in candidates if item.get("status") == "confirmed"]
    top_candidate = str((candidates[0] if candidates else {}).get("text", "") or "").strip()

    qwen_norm = normalize_text(qwen_candidate)
    if qwen_norm:
        joined_norm = normalize_text(_ocr_joined_text(ocr_result))
        qwen_tokens = [
            normalize_text(token)
            for token in re.findall(r"[A-Za-z0-9]+|[一-鿿]+", qwen_candidate)
        ]
        coverage = sum(bool(token) and token in joined_norm for token in qwen_tokens) / max(len(qwen_tokens), 1)
        matching = [
            item for item in confirmed
            if normalize_text(item.get("text", "")) in qwen_norm
            or qwen_norm in normalize_text(item.get("text", ""))
        ]
        if matching and coverage >= 0.6:
            return qwen_candidate.strip(), ["qwen", "ocr", "text_role_rules"], top_candidate

    for item in confirmed:
        text = str(item.get("text", "") or "").strip()
        if text:
            return text, ["ocr", "text_role_rules", "layout_geometry", "linguistic_gate"], top_candidate
    return "", [], top_candidate


def _resolve_brand_info(
    qwen_result: Dict[str, Any],
    ocr_result: Dict[str, Any],
    notes: List[str],
    text_roles: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    qwen_brand = _safe_dict(qwen_result.get("brand_info"))
    qwen_text = str(qwen_brand.get("brand_text", "") or "").strip()
    qwen_valid = bool(qwen_brand.get("has_brand")) and bool(qwen_text)
    qwen_text_analysis = _safe_dict(qwen_result.get("text_analysis"))

    ocr_candidate, ocr_sources, top_candidate = _select_ocr_brand_candidate(
        ocr_result,
        qwen_candidate=qwen_text,
        qwen_suspicious_text=_ensure_list(qwen_text_analysis.get("suspicious_text")),
        text_roles=text_roles,
    )

    qwen_supported = qwen_valid and ocr_candidate and (
        _normalize_text(qwen_text) == _normalize_text(ocr_candidate)
        or _normalize_text(qwen_text) in _normalize_text(ocr_candidate)
        or _normalize_text(ocr_candidate) in _normalize_text(qwen_text)
    )

    if qwen_supported:
        return (
            {"has_brand": True, "brand_text": qwen_text, "brand_type": "visible_brand", "brand_status": "confirmed", "brand_candidate": top_candidate, "confidence": 0.9},
            _source_record(
                value=qwen_text,
                sources=["qwen", "ocr", "text_role_rules"],
                inferred=False,
                evidence=_unique_strings(qwen_text, ocr_candidate),
            ),
        )

    if qwen_valid and not qwen_supported:
        notes.append(
            f"Qwen品牌候选“{qwen_text}”虽出现在OCR中，但未通过品牌/标题/CTA角色校验，已清空。"
        )

    if ocr_candidate:
        notes.append(f"采用通过文本角色校验的OCR品牌候选“{ocr_candidate}”。")
        return (
            {"has_brand": True, "brand_text": ocr_candidate, "brand_type": "visible_generated_brand" if any(is_watermark(x) for x in _ocr_lines(ocr_result)) else "visible_brand", "brand_status": "confirmed", "brand_candidate": top_candidate, "confidence": float(_safe_dict(text_roles).get("brand_confidence", 0.85) or 0.85)},
            _source_record(
                value=ocr_candidate,
                sources=ocr_sources,
                inferred=False,
                evidence=[ocr_candidate],
            ),
        )

    return (
        {"has_brand": False, "brand_text": "", "brand_type": "no_brand", "brand_status": "candidate" if top_candidate else "none", "brand_candidate": top_candidate, "confidence": 0.0},
        _source_record(value="", sources=[], inferred=False, evidence=[]),
    )


def _extract_event_time(text: str) -> str:
    normalized = (
        text.replace("：", ":")
        .replace("／", "/")
        .replace("－", "-")
    )

    patterns = (
        r"(20\d{2}[/-]\d{1,2}[/-]\d{1,2}\s*\d{1,2}:\d{2})",
        r"(20\d{2}[/-]\d{1,2}[/-]\d{1,2})",
        r"(\d{1,2}月\d{1,2}日\s*\d{1,2}:\d{2})",
        r"(\d{1,2}月\d{1,2}日)",
        r"(?<!\d)(\d{1,2}:\d{2})(?!\d)",
    )

    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            return re.sub(r"\s+", " ", match.group(1)).strip()

    return ""


def _extract_event_location(lines: List[str]) -> str:
    for line in lines:
        match = re.search(
            r"(?:活动地点|演出地点|比赛地点|地点)[：:\s]*([^，。；;]+)",
            line,
        )
        if match:
            return match.group(1).strip()
    return ""


def _event_title_score(
    line: str,
    brand_text: str,
) -> float:
    text = line.strip()

    if not text:
        return -100.0

    if brand_text and _normalize_text(text) == _normalize_text(brand_text):
        return -20.0

    if _is_domain(text):
        return -20.0

    if re.fullmatch(r"\d{1,6}", text):
        return -10.0

    if _contains_any(text, ("时间", "地点", "日期", "主办方", "承办方")):
        return -10.0

    score = 0.0

    if _contains_any(text, EVENT_TITLE_HINTS):
        score += 6.0

    if 4 <= len(text) <= 24:
        score += 2.0

    if re.search(r"[\u4e00-\u9fff]", text):
        score += 1.0

    return score


def _extract_event_name(
    ocr_result: Dict[str, Any],
    brand_text: str,
) -> str:
    lines = _ocr_lines(ocr_result)
    scored = [
        (_event_title_score(line, brand_text), line)
        for line in lines
    ]
    scored.sort(key=lambda item: item[0], reverse=True)

    if scored and scored[0][0] >= 6.0:
        return scored[0][1].strip()

    return ""


def _extract_organizer(lines: List[str]) -> str:
    for line in lines:
        match = re.search(
            r"(?:主办方|主办单位|承办方|承办单位)[：:\s]*([^，。；;]+)",
            line,
        )
        if match:
            return match.group(1).strip()
    return ""


def _resolve_event_info(
    qwen_result: Dict[str, Any],
    ocr_result: Dict[str, Any],
    scene_type: str,
    brand_info: Dict[str, Any],
    notes: List[str],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    raw = _safe_dict(qwen_result.get("event_info"))
    qwen_text_analysis = _safe_dict(qwen_result.get("text_analysis"))
    analysis = analyze_text_semantics(
        ocr_result,
        qwen_brand_candidate=str(_safe_dict(qwen_result.get("brand_info")).get("brand_text", "") or ""),
        qwen_suspicious_text=_ensure_list(qwen_text_analysis.get("suspicious_text")),
    )

    joined = _ocr_joined_text(ocr_result)
    qwen_values = {
        "event_name": str(raw.get("event_name", "") or "").strip(),
        "event_time": str(raw.get("event_time", "") or "").strip(),
        "event_location": str(raw.get("event_location", "") or "").strip(),
        "organizer": str(raw.get("organizer", "") or "").strip(),
    }

    dates = _unique_strings(analysis.get("event_dates"))
    times = _unique_strings(analysis.get("event_times"))
    locations = _unique_strings(analysis.get("event_locations"))
    event_time = " / ".join(_unique_strings(dates, times)[:3])
    event_location = locations[0] if locations else ""

    headline = str(qwen_text_analysis.get("headline", "") or "").strip()
    event_name = ""
    if qwen_values["event_name"] and _text_supported(qwen_values["event_name"], ocr_result):
        event_name = qwen_values["event_name"]
    elif scene_type in {"活动宣传海报", "教育校园宣传"} and headline and _text_supported(headline, ocr_result):
        event_name = headline

    organizer = ""
    if qwen_values["organizer"] and _text_supported(qwen_values["organizer"], ocr_result):
        organizer = qwen_values["organizer"]

    # Accept Qwen-formatted time/location only when the normalized value is OCR-supported.
    if not event_time and qwen_values["event_time"] and _text_supported(qwen_values["event_time"], ocr_result):
        event_time = qwen_values["event_time"]
    if not event_location and qwen_values["event_location"] and _text_supported(qwen_values["event_location"], ocr_result):
        event_location = qwen_values["event_location"]

    resolved = {
        "event_name": event_name,
        "event_time": event_time,
        "event_location": event_location,
        "organizer": organizer,
    }
    has_event = bool(
        scene_type == "活动宣传海报"
        or (scene_type == "教育校园宣传" and (event_time or event_location))
        or any(resolved.values())
    )

    provenance: Dict[str, Any] = {}
    for key, value in resolved.items():
        sources = []
        if value:
            sources.append("ocr")
        if value and value in qwen_values.values():
            sources.append("qwen")
        provenance[key] = _source_record(
            value=value,
            sources=sources,
            inferred=False,
            evidence=[value] if value else [],
        )
    provenance["has_event"] = _source_record(
        value=has_event,
        sources=["ocr", "rule_inference"] if has_event else [],
        inferred=has_event and not any(resolved.values()),
        evidence=[scene_type] if has_event else [],
    )

    for key, value in qwen_values.items():
        if value and not resolved.get(key):
            notes.append(f"Qwen给出的{key}“{value}”缺少OCR或文本角色支持，已清空。")

    return ({"has_event": has_event, **resolved}, provenance)


def _extract_campaign_words(
    ocr_result: Dict[str, Any],
    scene_type: str,
) -> List[str]:
    lines = _ocr_lines(ocr_result)
    joined = _ocr_joined_text(ocr_result)

    result: List[str] = []

    for hint in CAMPAIGN_HINTS:
        if hint in joined:
            result.append(hint)

    numeric_lines = [
        line
        for line in lines
        if re.fullmatch(r"\d{3,4}", line.strip())
    ]

    for token in numeric_lines:
        repeated = sum(1 for line in lines if line.strip() == token) >= 2
        if repeated or scene_type in {"活动宣传海报", "电商商品广告"}:
            result.append(token)

    return _unique_strings(result)


def _resolve_basic_facts(
    ocr_result: Dict[str, Any],
    qwen_result: Dict[str, Any],
    scene_type: str,
    notes: List[str],
    text_roles: Optional[Dict[str, Any]] = None,
) -> Tuple[
    Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]
]:
    """Hard textual facts are accepted only from deterministic OCR semantics."""
    qwen_text_analysis = _safe_dict(qwen_result.get("text_analysis"))
    analysis = analyze_text_semantics(
        ocr_result,
        qwen_brand_candidate=str(_safe_dict(qwen_result.get("brand_info")).get("brand_text", "") or ""),
        qwen_suspicious_text=_ensure_list(qwen_text_analysis.get("suspicious_text")),
    )

    role_data = _safe_dict(text_roles)
    prices = _unique_strings(role_data.get("price_text") or analysis.get("price_text"))
    promotions = _unique_strings(role_data.get("promotion_words") or analysis.get("promotion_words"))
    ctas = _unique_strings(role_data.get("cta_text") or analysis.get("cta_text"))
    cta_raw_text = _unique_strings(role_data.get("cta_raw_text"))
    cta_canonical = _unique_strings(role_data.get("cta_canonical"))
    cta_evidence = [
        item for item in (role_data.get("cta_evidence") or [])
        if isinstance(item, dict)
    ]

    price_info = {"has_price": bool(prices), "price_text": "，".join(prices)}
    promotion_info = {
        "has_promotion": bool(promotions),
        "promotion_words": promotions,
        "promotion_type": "explicit_offer" if promotions else "none",
    }
    cta_info = {
        "has_cta": bool(ctas),
        "cta_text": "，".join(ctas),
        "cta_raw_text": "，".join(cta_raw_text),
        "cta_canonical": cta_canonical,
        "evidence_source": "ocr" if ctas else "",
        "confidence": 0.96 if ctas and cta_raw_text else (0.9 if ctas else 0.0),
        "evidence": cta_evidence,
    }

    campaign_words = _extract_campaign_words(ocr_result, scene_type)
    campaign_info = {
        "has_campaign": bool(campaign_words),
        "campaign_words": campaign_words,
    }

    # Record rejected Qwen-only facts for auditability.
    qwen_price = str(_safe_dict(qwen_result.get("price_info")).get("price_text", "") or "").strip()
    if qwen_price and not prices:
        notes.append(f"Qwen价格“{qwen_price}”未通过OCR价格规则，已清空。")
    qwen_cta = str(_safe_dict(qwen_result.get("cta_info")).get("cta_text", "") or "").strip()
    if qwen_cta and not ctas:
        notes.append(f"Qwen CTA“{qwen_cta}”未通过OCR CTA规则，已清空。")
    qwen_promos = _unique_strings(_safe_dict(qwen_result.get("promotion_info")).get("promotion_words"))
    if qwen_promos and not promotions:
        notes.append("Qwen促销信息未通过OCR促销规则，已清空：" + "、".join(qwen_promos[:4]))

    return (
        price_info,
        promotion_info,
        campaign_info,
        cta_info,
        {
            "price_info": _source_record(price_info["price_text"], ["ocr", "text_role_rules"] if prices else [], evidence=prices),
            "promotion_info": _source_record(promotions, ["ocr", "text_role_rules"] if promotions else [], evidence=promotions),
            "campaign_info": _source_record(campaign_words, ["ocr"] if campaign_words else [], evidence=campaign_words),
            "cta_info": _source_record(
                ctas,
                ["ocr", "text_role_rules"] if ctas else [],
                evidence=cta_evidence or cta_raw_text or ctas,
            ),
        },
    )


def _semantic_deduplicate(
    values: Any,
    max_items: int = 8,
) -> List[str]:
    items = _unique_strings(values)
    result: List[str] = []

    for item in items:
        normalized = _normalize_text(item)
        if not normalized:
            continue

        duplicate = False
        for existing in result:
            existing_norm = _normalize_text(existing)

            if normalized == existing_norm:
                duplicate = True
                break

            # 只删除高度重叠的长短表述，避免普通短词误删。
            shorter = min(len(normalized), len(existing_norm))
            longer = max(len(normalized), len(existing_norm))
            if (
                shorter >= 5
                and shorter / max(longer, 1) >= 0.65
                and (
                    normalized in existing_norm
                    or existing_norm in normalized
                )
            ):
                duplicate = True
                break

        if not duplicate:
            result.append(item)

        if len(result) >= max_items:
            break

    return result


def _build_trust_categories(
    brand_info: Dict[str, Any],
    brand_provenance: Dict[str, Any],
    event_info: Dict[str, Any],
    event_provenance: Dict[str, Any],
    price_info: Dict[str, Any],
    price_provenance: Dict[str, Any],
    selling_points: List[str],
    trust_signals: List[str],
    ocr_result: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    categories: Dict[str, Dict[str, Any]] = {
        key: _new_category()
        for key in (
            "brand_identity",
            "official_channel",
            "event_time",
            "event_location",
            "organizer_identity",
            "authority_endorsement",
            "transparent_price",
            "concrete_information",
            "realistic_evidence",
        )
    }

    brand_text = str(brand_info.get("brand_text", "") or "").strip()
    if brand_info.get("has_brand") and brand_text:
        for source in brand_provenance.get("sources", []) or ["qwen"]:
            _add_category(
                categories,
                "brand_identity",
                brand_text,
                source,
                bool(brand_provenance.get("inferred")),
            )

    joined = _ocr_joined_text(ocr_result)
    domains = _extract_domains(joined)
    official_signals = [
        signal
        for signal in trust_signals
        if _contains_any(signal, OFFICIAL_CHANNEL_HINTS)
        or _is_domain(signal)
    ]
    if domains:
        for domain in domains:
            _add_category(
                categories,
                "official_channel",
                domain,
                "ocr",
            )
    for signal in official_signals:
        _add_category(
            categories,
            "official_channel",
            signal,
            "qwen",
        )

    event_time = str(event_info.get("event_time", "") or "").strip()
    if event_time:
        for source in event_provenance["event_time"].get("sources", []) or ["qwen"]:
            _add_category(
                categories,
                "event_time",
                event_time,
                source,
            )

    event_location = str(event_info.get("event_location", "") or "").strip()
    if event_location:
        for source in event_provenance["event_location"].get("sources", []) or ["qwen"]:
            _add_category(
                categories,
                "event_location",
                event_location,
                source,
            )

    organizer = str(event_info.get("organizer", "") or "").strip()
    if organizer:
        for source in event_provenance["organizer"].get("sources", []) or ["qwen"]:
            _add_category(
                categories,
                "organizer_identity",
                organizer,
                source,
            )

    for signal in trust_signals:
        if _contains_any(signal, AUTHORITY_HINTS):
            _add_category(
                categories,
                "authority_endorsement",
                signal,
                "qwen",
            )

        if _contains_any(signal, REALISTIC_EVIDENCE_HINTS):
            _add_category(
                categories,
                "realistic_evidence",
                signal,
                "qwen",
            )

    if price_info.get("has_price"):
        for source in price_provenance.get("sources", []) or ["qwen"]:
            _add_category(
                categories,
                "transparent_price",
                price_info.get("price_text", ""),
                source,
            )

    # “具体信息”只接受OCR可核验内容。纯视觉描述不用于抬高信任度。
    ocr_supported_points = [
        point for point in selling_points
        if _text_supported(point, ocr_result)
    ]
    if ocr_supported_points:
        _add_category(
            categories,
            "concrete_information",
            ocr_supported_points,
            "ocr",
        )

    return categories


def _build_memory_categories(
    brand_info: Dict[str, Any],
    brand_provenance: Dict[str, Any],
    event_info: Dict[str, Any],
    event_provenance: Dict[str, Any],
    main_subject: str,
    main_subject_source: str,
    memory_points: List[str],
    attention_elements: List[str],
    emotion_style: Dict[str, Any],
    emotion_source: str,
    ocr_result: Dict[str, Any],
    campaign_info: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    categories: Dict[str, Dict[str, Any]] = {
        key: _new_category()
        for key in (
            "brand",
            "event_or_slogan",
            "numeric_symbol",
            "distinctive_subject",
            "visual_symbol",
            "color_style",
        )
    }

    brand_text = str(brand_info.get("brand_text", "") or "").strip()
    if brand_info.get("has_brand") and brand_text:
        for source in brand_provenance.get("sources", []) or ["qwen"]:
            _add_category(
                categories,
                "brand",
                brand_text,
                source,
                bool(brand_provenance.get("inferred")),
            )

    event_name = str(event_info.get("event_name", "") or "").strip()
    if event_name:
        for source in event_provenance["event_name"].get("sources", []) or ["qwen"]:
            _add_category(
                categories,
                "event_or_slogan",
                event_name,
                source,
            )
    else:
        for point in memory_points:
            if _contains_any(point, EVENT_TITLE_HINTS):
                _add_category(
                    categories,
                    "event_or_slogan",
                    point,
                    "qwen",
                )

    # 数字符号：只使用独立数字标题或活动节点，不把日期时间作为数字记忆点。
    numeric_candidates = [
        line
        for line in _ocr_lines(ocr_result)
        if re.fullmatch(r"\d{2,4}", line.strip())
    ]
    numeric_candidates.extend(campaign_info.get("campaign_words", []))
    for candidate in _unique_strings(numeric_candidates):
        if re.fullmatch(r"\d{2,4}", candidate):
            _add_category(
                categories,
                "numeric_symbol",
                candidate,
                "ocr",
            )

    subject_norm = _normalize_text(main_subject)
    brand_norm = _normalize_text(brand_text)
    event_norm = _normalize_text(event_name)

    if (
        main_subject
        and subject_norm
        and subject_norm not in {brand_norm, event_norm}
    ):
        _add_category(
            categories,
            "distinctive_subject",
            main_subject,
            main_subject_source,
            main_subject_source == "rule_inference",
        )

    for item in _unique_strings(memory_points, attention_elements):
        # 大型数字属于numeric_symbol，不同时计入visual_symbol。
        if re.search(r"(?:大型|巨大|醒目|中央).{0,4}\d{2,4}", item):
            continue
        if _contains_any(item, VISUAL_SYMBOL_HINTS):
            _add_category(
                categories,
                "visual_symbol",
                item,
                "qwen",
            )
        if _contains_any(item, COLOR_STYLE_HINTS):
            _add_category(
                categories,
                "color_style",
                item,
                "visual" if item.startswith(("高饱和", "高对比")) else "qwen",
            )

    emotion_evidence = _unique_strings(
        _safe_dict(emotion_style).get("evidence")
    )
    for item in emotion_evidence:
        if _contains_any(item, COLOR_STYLE_HINTS):
            _add_category(
                categories,
                "color_style",
                item,
                emotion_source or "qwen",
            )

    return categories


def _categories_to_signals(
    categories: Dict[str, Dict[str, Any]],
    labels: Dict[str, str],
) -> List[str]:
    result: List[str] = []

    for key, label in labels.items():
        category = categories.get(key, {})
        if not category.get("present"):
            continue

        evidence = _unique_strings(category.get("evidence"))
        if evidence:
            result.append(f"{label}：" + "；".join(evidence))
        else:
            result.append(label)

    return result



def _resolve_scene_specific_subject_type(
    *,
    scene_type: str,
    ocr_result: Dict[str, Any],
    main_subject: str,
    product_type: str,
) -> Dict[str, Any]:
    """Normalize advertised object/type after scene resolution.

    Rules describe general semantic conflicts (format vs. product, event reward
    vs. price, campus competition vs. course) and never use file paths.
    """
    joined = _ocr_joined_text(ocr_result)
    compact = _normalize_text(joined)
    subject = str(main_subject or "").strip()
    product = str(product_type or "").strip()
    reasons: List[str] = []

    if scene_type == "短视频封面":
        entertainment = any(token in compact for token in (
            "mustwatch", "trendalert", "trending", "clicknow", "viral", "hot"
        ))
        new_product = "娱乐类短视频封面" if entertainment else "短视频内容"
        if product != new_product:
            product = new_product
            reasons.append("场景已确认为短视频封面，产品类型改为内容载体而非画面服饰/壁纸")

    elif scene_type == "活动宣传海报":
        competition = any(token in compact for token in (
            "hackathon", "challenge", "competition", "prizes", "register",
            "topteam", "teamsof", "比赛", "竞赛", "挑战"
        ))
        campus = any(token in compact for token in (
            "campus", "student", "engineeringhall", "university", "校园", "学生"
        ))
        if competition and campus:
            new_subject = "校园科技竞赛" if any(token in compact for token in ("hackathon", "codebuild", "innovationlab")) else "校园比赛"
            new_product = "校园比赛宣传海报"
            if subject != new_subject or product != new_product:
                subject, product = new_subject, new_product
                reasons.append("检测到校园、参赛、奖项和报名信息，修正为校园竞赛宣传")

    elif scene_type == "旅游宣传":
        if product in {"旅行摄影", "摄影", "城市风光", "旅行宣传"} or not product:
            product = "旅游宣传"
            reasons.append("旅游场景中将摄影/城市风光统一为旅游宣传内容")

    elif scene_type == "科技创新宣传":
        if product in {"科技", "未来科技", "科技产品", ""}:
            product = "科研项目宣传内容"
            reasons.append("科研项目海报中将泛化科技词规范为科研项目宣传内容")

    # Conservative product-category cleanup independent of scene folders.
    if any(token in compact for token in ("eaudeparfum", "parfum", "perfume")) and product in {"化妆品", "美妆产品", "美容产品", ""}:
        product = "香水"
        reasons.append("OCR出现香水品类术语，修正通用化妆品类型")

    # Qwen may hallucinate a specific model/brand in product_type. If the
    # specific string is absent from OCR, retain only the generic category.
    if "轿车" in product and not _text_supported(product, ocr_result):
        product = "汽车"
        reasons.append("具体汽车型号缺少OCR支持，回退为通用汽车类别")

    return {
        "main_subject": subject,
        "product_type": product,
        "changed": subject != str(main_subject or "").strip() or product != str(product_type or "").strip(),
        "source": "scene_semantic_resolution",
        "reasons": reasons,
    }

def validate_evidence(
    ocr_result: Dict[str, Any],
    visual_result: Dict[str, Any],
    qwen_result: Dict[str, Any],
    sam_result: Optional[Dict[str, Any]] = None,
    image_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    融合并校验OCR、Qwen、视觉特征和SAM证据。

    返回结构兼容旧版本，同时新增：
    - campaign_info
    - trust_categories
    - memory_categories
    - field_provenance
    """
    ocr_result = ocr_result or {}
    visual_result = visual_result or {}
    qwen_result = qwen_result or {}
    sam_result = sam_result or {}

    enriched = enrich_evidence(
        ocr_result,
        visual_result,
        qwen_result,
    ) or {}

    notes: List[str] = []
    qwen_text_analysis = _safe_dict(qwen_result.get("text_analysis"))
    text_analysis = analyze_text_semantics(
        ocr_result,
        qwen_brand_candidate=str(_safe_dict(qwen_result.get("brand_info")).get("brand_text", "") or ""),
        qwen_suspicious_text=_ensure_list(qwen_text_analysis.get("suspicious_text")),
    )

    scene_type, scene_reasons, scene_provenance = _resolve_scene(
        qwen_result=qwen_result,
        ocr_result=ocr_result,
    )

    text_roles = classify_text_roles(
        ocr_result=ocr_result,
        scene_type=scene_type,
        image_size=_safe_dict(visual_result.get("image_size")),
    )

    brand_info, brand_provenance = _resolve_brand_info(
        qwen_result=qwen_result,
        ocr_result=ocr_result,
        notes=notes,
        text_roles=text_roles,
    )

    (
        price_info,
        promotion_info,
        campaign_info,
        cta_info,
        basic_provenance,
    ) = _resolve_basic_facts(
        ocr_result=ocr_result,
        qwen_result=qwen_result,
        scene_type=scene_type,
        notes=notes,
        text_roles=text_roles,
    )

    event_info, event_provenance = _resolve_event_info(
        qwen_result=qwen_result,
        ocr_result=ocr_result,
        scene_type=scene_type,
        brand_info=brand_info,
        notes=notes,
    )

    product_type = str(
        qwen_result.get("product_type", "") or ""
    ).strip()
    if product_type in SCENE_NAMES:
        notes.append(
            f"product_type“{product_type}”属于场景名称，已清空以避免字段混用。"
        )
        product_type = ""

    qwen_main_subject = str(
        qwen_result.get("main_subject", "") or ""
    ).strip()
    main_subject = qwen_main_subject
    main_subject_source = "qwen" if main_subject else ""

    brand_text = str(brand_info.get("brand_text", "") or "").strip()
    event_name = str(event_info.get("event_name", "") or "").strip()

    # 若具体品牌/型号没有通过事实门控，则将主对象回退为通用品类。
    raw_qwen_brand = str(
        _safe_dict(qwen_result.get("brand_info")).get("brand_text", "") or ""
    ).strip()
    if product_type and main_subject and not brand_text:
        model_like = bool(re.search(r"[A-Za-z]+\s*\d", main_subject))
        contains_removed_brand = bool(raw_qwen_brand) and (
            _normalize_text(raw_qwen_brand) in _normalize_text(main_subject)
        )
        if model_like or contains_removed_brand:
            notes.append(
                f"主对象“{main_subject}”包含未验证品牌/型号，已回退为“{product_type}”。"
            )
            main_subject = product_type
            main_subject_source = "rule_inference"

    # 活动场景中，若main_subject只是品牌，而存在明确活动名，则活动名为主对象。
    if (
        scene_type == "活动宣传海报"
        and event_name
        and brand_text
        and _normalize_text(main_subject) == _normalize_text(brand_text)
    ):
        main_subject = event_name
        main_subject_source = "rule_inference"
        notes.append(
            "活动场景中Qwen将品牌识别为主对象，已用明确活动名称修正主对象。"
        )

    # 跨模态主体对齐：OCR负责产品/服务语义，Qwen负责可见对象。
    # 仅在语义证据强且原主体更像人物、容器、设备或陪衬时修正，
    # 不读取文件夹名，不补具体品牌/型号。
    subject_alignment = align_main_subject(
        ocr_result=ocr_result,
        qwen_result=qwen_result,
        scene_type=scene_type,
        current_subject=main_subject,
        current_product_type=product_type,
        text_analysis=text_analysis,
    )
    if subject_alignment.get("changed"):
        old_subject = main_subject
        old_product = product_type
        main_subject = str(subject_alignment.get("main_subject", main_subject) or main_subject).strip()
        product_type = str(subject_alignment.get("product_type", product_type) or product_type).strip()
        main_subject_source = str(subject_alignment.get("source", "cross_modal_alignment") or "cross_modal_alignment")
        notes.append(
            "跨模态主体对齐："
            f"“{old_subject or '空'} / {old_product or '空'}” → "
            f"“{main_subject or '空'} / {product_type or '空'}”。"
        )

    format_resolution = _resolve_scene_specific_subject_type(
        scene_type=scene_type,
        ocr_result=ocr_result,
        main_subject=main_subject,
        product_type=product_type,
    )
    if format_resolution.get("changed"):
        old_subject, old_product = main_subject, product_type
        main_subject = str(format_resolution.get("main_subject", main_subject) or main_subject).strip()
        product_type = str(format_resolution.get("product_type", product_type) or product_type).strip()
        main_subject_source = "scene_semantic_resolution"
        notes.append(
            "场景语义字段修正："
            f"“{old_subject or '空'} / {old_product or '空'}” → "
            f"“{main_subject or '空'} / {product_type or '空'}”。"
        )

    subject_resolution = resolve_subjects(
        qwen_result=qwen_result,
        advertised_subject=main_subject,
        product_type=product_type,
        scene_type=scene_type,
        image_path=image_path,
    )
    visual_subject = str(subject_resolution.get("visual_subject", "") or "").strip()
    advertised_subject = str(subject_resolution.get("advertised_subject", main_subject) or main_subject).strip()

    target_audience = str(
        qwen_result.get("target_audience", "") or ""
    ).strip()

    raw_selling_points = _semantic_deduplicate(
        _unique_strings(
            enriched.get("selling_points"),
            qwen_result.get("visible_evidence"),
            qwen_result.get("selling_points"),
        ),
        max_items=10,
    )
    selling_points = _filter_visual_or_ocr_evidence(
        raw_selling_points,
        ocr_result=ocr_result,
        notes=notes,
        field_name="卖点",
        max_items=6,
    )
    selling_points = _filter_unsupported_color_claims(
        selling_points,
        image_path=image_path,
        notes=notes,
        field_name="卖点",
    )

    raw_trust_candidates = _semantic_deduplicate(
        _unique_strings(
            enriched.get("trust_signals"),
            qwen_result.get("trust_signals"),
        ),
        max_items=10,
    )
    raw_trust_candidates = _filter_textual_elements_without_text(
        raw_trust_candidates,
        ocr_result=ocr_result,
        notes=notes,
        field_name="信任证据",
        max_items=8,
    )
    raw_trust_signals = _filter_visual_or_ocr_evidence(
        raw_trust_candidates,
        ocr_result=ocr_result,
        notes=notes,
        field_name="信任证据",
        max_items=8,
    )

    attention_candidates = _semantic_deduplicate(
        _unique_strings(
            enriched.get("attention_elements"),
            qwen_result.get("attention_elements"),
        ),
        max_items=10,
    )
    attention_candidates = _filter_textual_elements_without_text(
        attention_candidates,
        ocr_result=ocr_result,
        notes=notes,
        field_name="注意力证据",
        max_items=8,
    )
    attention_elements = _filter_visual_or_ocr_evidence(
        attention_candidates,
        ocr_result=ocr_result,
        notes=notes,
        field_name="注意力证据",
        max_items=6,
    )

    memory_candidates = _semantic_deduplicate(
        _unique_strings(
            enriched.get("memory_points"),
            qwen_result.get("memory_points"),
            qwen_result.get("visible_evidence"),
            selling_points,
        ),
        max_items=10,
    )
    memory_candidates = _filter_textual_elements_without_text(
        memory_candidates,
        ocr_result=ocr_result,
        notes=notes,
        field_name="记忆证据",
        max_items=8,
    )
    memory_points = _filter_visual_or_ocr_evidence(
        memory_candidates,
        ocr_result=ocr_result,
        notes=notes,
        field_name="记忆证据",
        max_items=8,
    )
    memory_points = _filter_unsupported_color_claims(
        memory_points,
        image_path=image_path,
        notes=notes,
        field_name="记忆证据",
    )

    risk_points = _semantic_deduplicate(
        _filter_textual_elements_without_text(
            _unique_strings(
                enriched.get("risk_points"),
                qwen_result.get("risk_points"),
            ),
            ocr_result=ocr_result,
            notes=notes,
            field_name="风险证据",
            max_items=6,
        ),
        max_items=6,
    )

    if text_analysis.get("status") in {"partially_readable", "garbled"}:
        suspicious = _unique_strings(text_analysis.get("suspicious_lines"))
        if suspicious:
            risk_points = _semantic_deduplicate(
                _unique_strings(
                    risk_points,
                    ["疑似错字或乱码：" + "、".join(suspicious[:4])],
                ),
                max_items=6,
            )
            notes.append(
                f"文字状态判定为{text_analysis.get('status')}，已记录乱码风险。"
            )

    fallback_emotion, fallback_attention, fallback_memory = (
        _visual_feature_fallbacks(visual_result)
    )

    qwen_emotion = _safe_dict(qwen_result.get("emotion_style"))
    enriched_emotion = _safe_dict(enriched.get("emotion_style"))
    qwen_emotion_evidence = _filter_visual_or_ocr_evidence(
        qwen_emotion.get("evidence"),
        ocr_result=ocr_result,
        max_items=4,
    )
    enriched_emotion_evidence = _filter_visual_or_ocr_evidence(
        enriched_emotion.get("evidence"),
        ocr_result=ocr_result,
        max_items=4,
    )

    if qwen_emotion_evidence:
        emotion_style = {
            "main_emotion": _normalize_emotion_label(
                qwen_emotion.get("main_emotion", "其他")
            ),
            "evidence": qwen_emotion_evidence,
        }
        emotion_source = "qwen"
    elif enriched_emotion_evidence:
        emotion_style = {
            "main_emotion": _normalize_emotion_label(
                enriched_emotion.get("main_emotion", "其他")
            ),
            "evidence": enriched_emotion_evidence,
        }
        emotion_source = "rule_inference"
    else:
        emotion_style = fallback_emotion
        emotion_source = "visual"
        notes.append("情绪语义缺少有效证据，已采用基础视觉特征生成可解释兜底。")

    # 旅游视觉的心理语义不能仅由饱和度推断。仅在Qwen未给出有效
    # 情绪证据时，使用已确认的旅游场景和可见主体做低强度语义兜底。
    if scene_type == "旅游宣传" and emotion_source == "visual":
        travel_text = " ".join(
            _unique_strings(
                main_subject,
                qwen_result.get("scene_evidence"),
                qwen_result.get("scene_subtype"),
            )
        )
        if _contains_any(
            travel_text,
            ("海滩", "海岛", "沙滩", "海景", "度假", "酒店", "度假村", "泳池", "棕榈", "椰树", "海边"),
        ):
            emotion_style = {
                "main_emotion": "温暖愉悦",
                "evidence": _unique_strings(
                    ["海岛、海滩或度假环境形成放松愉悦的场景语义"],
                    fallback_emotion.get("evidence"),
                    max_items=3,
                ),
            }
            emotion_source = "rule_inference"
        elif _contains_any(
            travel_text,
            ("地标", "城市风光", "景区", "景点", "目的地", "文化城市"),
        ):
            emotion_style = {
                "main_emotion": "温暖愉悦",
                "evidence": _unique_strings(
                    ["旅行目的地与城市景观形成探索和向往感"],
                    fallback_emotion.get("evidence"),
                    max_items=3,
                ),
            }
            emotion_source = "rule_inference"

    if len(attention_elements) < 2:
        attention_elements = _semantic_deduplicate(
            _unique_strings(attention_elements, fallback_attention),
            max_items=6,
        )
    # 视觉基础特征产生的色彩/光影记忆证据始终参与融合，
    # 防止已有两个语义点时把颜色风格完全漏掉。
    memory_points = _semantic_deduplicate(
        _unique_strings(memory_points, fallback_memory),
        max_items=8,
    )

    trust_categories = _build_trust_categories(
        brand_info=brand_info,
        brand_provenance=brand_provenance,
        event_info=event_info,
        event_provenance=event_provenance,
        price_info=price_info,
        price_provenance=basic_provenance["price_info"],
        selling_points=selling_points,
        trust_signals=raw_trust_signals,
        ocr_result=ocr_result,
    )

    memory_categories = _build_memory_categories(
        brand_info=brand_info,
        brand_provenance=brand_provenance,
        event_info=event_info,
        event_provenance=event_provenance,
        main_subject=main_subject,
        main_subject_source=main_subject_source or "qwen",
        memory_points=memory_points,
        attention_elements=attention_elements,
        emotion_style=emotion_style,
        emotion_source=emotion_source,
        ocr_result=ocr_result,
        campaign_info=campaign_info,
    )

    # 用类别生成规范化解释文本。原始信号仍保留在raw_trust_signals中。
    trust_signals = _categories_to_signals(
        trust_categories,
        {
            "brand_identity": "品牌身份明确",
            "official_channel": "存在官方渠道",
            "event_time": "活动时间明确",
            "event_location": "活动地点明确",
            "organizer_identity": "主办或承办主体明确",
            "authority_endorsement": "存在权威或合作背书",
            "transparent_price": "价格信息透明",
            "concrete_information": "存在具体信息",
            "realistic_evidence": "存在真实或可核验依据",
        },
    )

    # 规范化记忆点，只保留类别对应的代表证据，避免相同事实重复计分。
    canonical_memory_points = _categories_to_signals(
        memory_categories,
        {
            "brand": "品牌记忆点",
            "event_or_slogan": "活动名或口号记忆点",
            "numeric_symbol": "数字记忆点",
            "distinctive_subject": "独特主体记忆点",
            "visual_symbol": "视觉符号记忆点",
            "color_style": "色彩风格记忆点",
        },
    )

    confidence = _safe_dict(enriched.get("confidence"))
    if not confidence:
        confidence = _safe_dict(qwen_result.get("confidence"))

    scene_subtype = str(
        qwen_result.get("scene_subtype", "") or ""
    ).strip()
    if (
        scene_type == "品牌广告"
        and not _has_meaningful_ocr_text(ocr_result)
        and main_subject
    ):
        scene_subtype = "无文字产品主视觉"

    scene_evidence = _unique_strings(
        qwen_result.get("scene_evidence"),
        scene_reasons,
        [f"画面核心主体：{main_subject}"] if main_subject else [],
    )

    field_provenance = {
        "scene_type": scene_provenance,
        "main_subject": _source_record(
            value=main_subject,
            sources=[main_subject_source] if main_subject_source else [],
            inferred=main_subject_source in {"rule_inference", "cross_modal_alignment"},
            evidence=[main_subject] if main_subject else [],
        ),
        "visual_subject": _source_record(
            value=visual_subject,
            sources=[str(subject_resolution.get("source", ""))] if visual_subject else [],
            inferred=bool(subject_resolution.get("changed")),
            evidence=_unique_strings(subject_resolution.get("reasons"), visual_subject),
        ),
        "advertised_subject": _source_record(
            value=advertised_subject,
            sources=[main_subject_source] if advertised_subject else [],
            inferred=main_subject_source in {"rule_inference", "cross_modal_alignment"},
            evidence=[advertised_subject] if advertised_subject else [],
        ),
        "brand_info": brand_provenance,
        "event_info": event_provenance,
        "selling_points": _source_record(
            value=selling_points,
            sources=["qwen"] if selling_points else [],
            inferred=False,
            evidence=selling_points,
        ),
        "emotion_style": _source_record(
            value=emotion_style,
            sources=[emotion_source] if emotion_source else [],
            inferred=emotion_source == "rule_inference",
            evidence=emotion_style.get("evidence", []),
        ),
        "attention_elements": _source_record(
            value=attention_elements,
            sources=["qwen", "visual"] if attention_elements else [],
            inferred=False,
            evidence=attention_elements,
        ),
        "text_analysis": _source_record(
            value=text_analysis,
            sources=["ocr", "qwen", "text_role_rules"],
            inferred=False,
            evidence=_unique_strings(text_analysis.get("suspicious_lines")),
        ),
        "memory_points": _source_record(
            value=memory_points,
            sources=["qwen", "visual"] if memory_points else [],
            inferred=False,
            evidence=memory_points,
        ),
        **basic_provenance,
    }

    return {
        "scene_type": scene_type,
        "scene_subtype": scene_subtype,
        "scene_evidence": scene_evidence,
        "scene_reasons": scene_reasons,
        "main_subject": advertised_subject,
        "visual_subject": visual_subject,
        "advertised_subject": advertised_subject,
        "product_type": product_type,
        "subject_alignment": subject_alignment,
        "subject_resolution": subject_resolution,
        "format_resolution": format_resolution,
        "target_audience": target_audience,
        "selling_points": selling_points,
        "price_info": price_info,
        "promotion_info": promotion_info,
        "campaign_info": campaign_info,
        "cta_info": cta_info,
        "brand_info": brand_info,
        "event_info": event_info,
        "text_analysis": text_analysis,
        "text_roles": text_roles,
        "visible_evidence": _unique_strings(qwen_result.get("visible_evidence")),
        "semantic_inference": _unique_strings(qwen_result.get("semantic_inference")),
        "emotion_style": emotion_style,
        "trust_signals": trust_signals,
        "raw_trust_signals": raw_trust_signals,
        "trust_categories": trust_categories,
        "attention_elements": attention_elements,
        "memory_points": canonical_memory_points,
        "raw_memory_points": memory_points,
        "memory_categories": memory_categories,
        "risk_points": risk_points,
        "visual_features": visual_result,
        "sam_features": sam_result,
        "ocr_result": ocr_result,
        "confidence": confidence,
        "field_provenance": field_provenance,
        "validator_notes": notes,
        "evidence_sources": {
            "qwen_semantics": bool(qwen_result)
            and not bool(qwen_result.get("fallback")),
            "ocr_factual_enrichment": bool(ocr_result),
            "visual_features": bool(visual_result),
            "sam": bool(sam_result),
            "scene_classifier": True,
            "category_deduplication": True,
            "source_tracking": True,
            "bilingual_text_semantics": True,
            "text_quality_analysis": True,
            "text_role_classification": True,
            "dual_subject_tracking": True,
            "opencv_face_presence": bool(_safe_dict(subject_resolution.get("face_evidence")).get("available")),
        },
    }