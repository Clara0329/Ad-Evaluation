"""
Qwen2-VL dual-view visible-evidence extractor V5.4.

用途：
1. Qwen2-VL只提取广告语义证据，不直接评分。
2. 先进行场景分类，再按场景提取品牌、活动、商品、情绪等证据。
3. 强制输出可解析JSON，并清洗嵌套列表、模板回显和异常字段。
4. OCR读取原图；Qwen读取文字遮罩图，避免把视觉理解退化为文字抄录。
5. 使用助手前缀和单行定长协议，降低小模型格式漂移。
6. 解析失败时不执行第二次多模态推理，直接启用本地兜底。
7. 自动优先加载项目models目录中的本地Qwen模型。
"""

from __future__ import annotations

import copy
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from modules.text_semantics import analyze_text_semantics


_model = None
_processor = None
_loaded_model_name: Optional[str] = None

DEFAULT_HF_MODEL = "Qwen/Qwen2-VL-2B-Instruct"

SCENE_TYPES = {
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
    "商品广告": "电商商品广告",
    "广告海报": "品牌广告",
    "品牌海报": "品牌广告",
    "活动海报": "活动宣传海报",
    "校园活动": "教育校园宣传",
    "教育培训": "教育校园宣传",
    "公益海报": "公益宣传",
    "旅游海报": "旅游宣传",
    "旅游宣传海报": "旅游宣传",
    "旅游广告": "旅游宣传",
    "度假宣传": "旅游宣传",
    "酒店推广": "旅游宣传",
    "度假村推广": "旅游宣传",
}

PLACEHOLDER_TEXTS = {
    "电商商品广告/品牌广告/活动宣传海报/短视频封面/教育校园宣传/公益宣传/旅游宣传/其他",
    "画面最主要的商品、人物或内容对象",
    "商品或内容类型；没有则写空字符串",
    "可能面向的人群；不确定则写空字符串",
    "画面中明确表达的卖点",
    "用于进一步说明具体场景",
    "判断场景所依据的文字或视觉证据",
    "品牌露出/参数说明/真实场景/用户评价/权威背书/质感真实等",
    "最吸引注意的视觉元素",
    "容易被记住的元素",
    "可能影响广告效果的问题",
}

STRUCTURED_PROMPT_TEMPLATE = """
你是广告图片的纯视觉语义提取器。输入图片中的文字区域已经被遮除；OCR文字由另一个模块处理。

只分析仍然可见的物体、场景、人物、构图、色彩、光影、材质和视觉氛围。禁止猜测、复述或补全任何品牌、标题、价格、优惠、CTA、网址、型号和广告文案。

只输出一行，严格使用下面格式，正好9个字段，字段之间必须使用双竖线 ||：
SCENE=<场景> || SUBJECT=<核心视觉主体> || PRODUCT=<产品或内容类型> || EVIDENCE=<证据1;证据2;证据3> || SELLING=<可见卖点1;可见卖点2> || EMOTION=<主要情绪> || ATTENTION=<元素1;元素2> || MEMORY=<元素1;元素2> || RISK=<风险1;风险2>

约束：
1. SCENE只能取：电商商品广告、品牌广告、活动宣传海报、短视频封面、教育校园宣传、公益宣传、旅游宣传、其他。
2. SUBJECT必须是画面面积最大或视觉中心的核心商品/场景，不能选择杯子、豆子、植物、图标、文字、背景人物等陪衬。
3. EVIDENCE只写直接可见的外观、材质、构图、光影、色彩或场景，每项不超过8个汉字。
4. SELLING只写可见视觉卖点，禁止推测性能、功效、舒适、耐磨、保湿、续航等不可见功能。
5. 列表内部使用英文分号 ; 分隔；没有内容写NONE。
6. 不要输出第二行、Markdown、JSON、解释、编号或示例。
"""

RETRY_PROMPT_TEMPLATE = ""  # v1.3.4继续禁止第二次完整多模态推理



def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _is_model_directory(path: Path) -> bool:
    if not path.is_dir() or not (path / "config.json").is_file():
        return False

    weight_names = (
        "model.safetensors",
        "pytorch_model.bin",
        "model-00001-of-00002.safetensors",
    )
    return any((path / name).exists() for name in weight_names) or any(
        path.glob("model-*.safetensors")
    )


def _resolve_model_name(model_name: Optional[str]) -> str:
    """优先顺序：显式参数 > 环境变量 > 项目本地模型 > HuggingFace名称。"""
    if model_name:
        return str(Path(model_name).expanduser()) if Path(model_name).expanduser().exists() else model_name

    env_path = os.environ.get("QWEN_MODEL_PATH", "").strip()
    if env_path:
        candidate = Path(env_path).expanduser()
        if _is_model_directory(candidate):
            return str(candidate.resolve())
        raise FileNotFoundError(f"QWEN_MODEL_PATH不是有效模型目录：{candidate}")

    project_models = _project_root() / "models"
    candidates = [
        project_models / "Qwen2-VL-2B-Instruct",
        project_models / "models--Qwen--Qwen2-VL-2B-Instruct",
    ]

    for candidate in candidates:
        if _is_model_directory(candidate):
            return str(candidate.resolve())

        snapshots_dir = candidate / "snapshots"
        if snapshots_dir.is_dir():
            snapshots = sorted(
                (p for p in snapshots_dir.iterdir() if _is_model_directory(p)),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if snapshots:
                return str(snapshots[0].resolve())

    return DEFAULT_HF_MODEL


def _get_model(model_name: Optional[str] = None):
    global _model, _processor, _loaded_model_name

    resolved_model_name = _resolve_model_name(model_name)
    if (
        _model is not None
        and _processor is not None
        and _loaded_model_name == resolved_model_name
    ):
        return _model, _processor, resolved_model_name

    import torch
    import transformers
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    major_version = int(transformers.__version__.split(".")[0])
    if major_version >= 5:
        raise RuntimeError(
            f"当前 transformers={transformers.__version__}，本项目要求4.x；"
            "建议安装 transformers==4.51.3。"
        )

    use_cuda = torch.cuda.is_available()
    local_only = Path(resolved_model_name).exists()

    _model = Qwen2VLForConditionalGeneration.from_pretrained(
        resolved_model_name,
        torch_dtype="auto" if use_cuda else torch.float32,
        device_map="auto" if use_cuda else None,
        low_cpu_mem_usage=True,
        local_files_only=local_only,
    )
    if not use_cuda:
        _model = _model.to("cpu")
    _model.eval()

    _processor = AutoProcessor.from_pretrained(
        resolved_model_name,
        min_pixels=256 * 28 * 28,
        max_pixels=512 * 28 * 28,
        use_fast=False,
        local_files_only=local_only,
    )
    _loaded_model_name = resolved_model_name
    return _model, _processor, resolved_model_name


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _find_balanced_json_object(text: str) -> Optional[str]:
    """从混杂文本中提取第一个括号平衡的JSON对象。"""
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False

    for index in range(start, len(text)):
        char = text[index]

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    return None


def _try_load_json(candidate: str) -> Optional[Dict[str, Any]]:
    candidate = candidate.strip()
    if not candidate:
        return None

    variants = [candidate]
    repaired = (
        candidate.replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
    )
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    if repaired != candidate:
        variants.append(repaired)

    for value in variants:
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _extract_json(text: str) -> Dict[str, Any]:
    if not text:
        return {"raw_text": "", "parse_error": True}

    cleaned = _strip_code_fences(text)

    parsed = _try_load_json(cleaned)
    if parsed is not None:
        return parsed

    balanced = _find_balanced_json_object(cleaned)
    if balanced:
        parsed = _try_load_json(balanced)
        if parsed is not None:
            return parsed

    return {"raw_text": cleaned, "parse_error": True}


def _clean_string(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    value = re.sub(r"\s+", " ", value).strip()
    return "" if value in PLACEHOLDER_TEXTS else value


def _flatten_strings(value: Any) -> List[str]:
    """递归提取模型错误嵌套结构中的可用字符串。"""
    result: List[str] = []

    if isinstance(value, str):
        cleaned = _clean_string(value)
        if cleaned:
            result.append(cleaned)
        return result

    if isinstance(value, list):
        for item in value:
            result.extend(_flatten_strings(item))
        return result

    if isinstance(value, dict):
        preferred_keys = ("description", "text", "content", "value", "name")
        for key in preferred_keys:
            if key in value:
                result.extend(_flatten_strings(value[key]))
                if result:
                    return result

        if "evidence" in value:
            result.extend(_flatten_strings(value["evidence"]))
        return result

    return result


def _clean_list(value: Any, max_items: int = 8) -> List[str]:
    result: List[str] = []
    for item in _flatten_strings(value):
        if item and item not in result:
            result.append(item)
        if len(result) >= max_items:
            break
    return result


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "是", "有"}
    return False


def _to_confidence(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return round(max(0.0, min(1.0, number)), 4)


def _normalized_match(value: str, text: str) -> bool:
    from modules.text_semantics import normalize_text
    value_norm = normalize_text(value)
    text_norm = normalize_text(text)
    return bool(value_norm) and bool(text_norm) and value_norm in text_norm


def _is_direct_visual_claim(value: str) -> bool:
    """判断卖点是否主要描述可见视觉事实，而非产品功能推测。"""
    text = str(value or "").strip()
    visual_terms = (
        "色", "构图", "光", "背景", "质感", "造型", "外观", "轮廓",
        "主体", "居中", "对比", "渐变", "光轨", "动感", "运动感",
        "极简", "简洁", "金属", "透明", "悬浮", "摄影", "场景",
        "纹理", "材质外观", "视觉", "明暗", "阴影", "高饱和", "低饱和",
        "屏幕", "镜头", "瓶身", "包装", "杯体", "鞋底", "车身",
        "轮毂", "人物", "表情", "姿态", "图标", "logo", "大型", "中央",
    )
    return any(term in text for term in visual_terms)


def _is_functional_claim(value: str) -> bool:
    text = str(value or "").strip()
    functional_terms = (
        "舒适", "耐磨", "防滑", "节能", "高效", "高性能", "处理器",
        "夜景模式", "保湿", "补水", "续航", "降噪", "音质", "功效",
        "效果好", "便携实用",
    )
    return any(term in text for term in functional_terms)


def _apply_ocr_fact_gates(
    cleaned: Dict[str, Any],
    ocr_result: Dict[str, Any],
) -> Dict[str, Any]:
    """对Qwen输出执行保守事实门控，避免无文字图片产生品牌/价格/活动幻觉。"""
    joined = str(ocr_result.get("joined_text", "") or "").strip()
    all_text = ocr_result.get("all_text", []) or []
    has_meaningful_text = any(len(str(item).strip()) >= 2 for item in all_text)
    qwen_text_analysis = cleaned.get("text_analysis") if isinstance(cleaned.get("text_analysis"), dict) else {}
    deterministic = analyze_text_semantics(
        ocr_result,
        qwen_brand_candidate=str((cleaned.get("brand_info") or {}).get("brand_text", "")) if isinstance(cleaned.get("brand_info"), dict) else "",
        qwen_suspicious_text=qwen_text_analysis.get("suspicious_text", []),
    )

    if not deterministic.get("price_text"):
        cleaned["price_info"] = {"has_price": False, "price_text": ""}

    if not deterministic.get("promotion_words"):
        cleaned["promotion_info"] = {
            "has_promotion": False,
            "promotion_words": [],
        }

    if not deterministic.get("cta_text"):
        cleaned["cta_info"] = {"has_cta": False, "cta_text": ""}

    brand = cleaned.get("brand_info", {})
    brand_text = str(brand.get("brand_text", "") or "").strip()
    valid_brand_candidates = [
        str(item.get("text", ""))
        for item in deterministic.get("brand_candidates", [])
        if isinstance(item, dict)
    ]
    brand_supported = bool(brand_text) and any(
        _normalized_match(brand_text, candidate)
        or _normalized_match(candidate, brand_text)
        for candidate in valid_brand_candidates
    )
    if not brand_supported:
        cleaned["brand_info"] = {"has_brand": False, "brand_text": ""}

    event = cleaned.get("event_info", {})
    event_fields = {
        key: str(event.get(key, "") or "").strip()
        for key in ("event_name", "event_time", "event_location", "organizer")
    }
    for key, value in list(event_fields.items()):
        if not value or not _normalized_match(value, joined):
            event_fields[key] = ""
    has_event = bool(any(event_fields.values()))
    cleaned["event_info"] = {"has_event": has_event, **event_fields}

    original_brand = brand_text
    subject = str(cleaned.get("main_subject", "") or "").strip()
    product_type = str(cleaned.get("product_type", "") or "").strip()
    if product_type and subject and not brand_supported:
        looks_model_specific = bool(re.search(r"[A-Za-z]+\s*\d", subject))
        mentions_removed_brand = bool(original_brand) and _normalized_match(original_brand, subject)
        if looks_model_specific or mentions_removed_brand:
            cleaned["main_subject"] = product_type

    for field, limit in (
        ("selling_points", 6),
        ("attention_elements", 5),
        ("memory_points", 5),
        ("trust_signals", 6),
    ):
        filtered = []
        for item in cleaned.get(field, []) or []:
            item_text = str(item or "").strip()
            if not item_text:
                continue
            text_supported = _normalized_match(item_text, joined)
            if _is_functional_claim(item_text) and not text_supported:
                continue
            if text_supported or _is_direct_visual_claim(item_text):
                filtered.append(item_text)
        cleaned[field] = filtered[:limit]

    textual_terms = ("标题", "文案", "参数", "价格", "文字", "信息区", "按钮")
    if not has_meaningful_text:
        for field in ("attention_elements", "memory_points", "trust_signals", "risk_points"):
            cleaned[field] = [
                item
                for item in (cleaned.get(field, []) or [])
                if not any(term in str(item) for term in textual_terms)
            ]

    cleaned["text_analysis"] = {
        "status": deterministic.get("status", "absent"),
        "headline": str(qwen_text_analysis.get("headline", "") or "").strip(),
        "brand_candidate": brand_text if brand_supported else "",
        "suspicious_text": deterministic.get("suspicious_lines", []),
    }
    return cleaned



def _normalize_scene_type(value: Any, ocr_result: Dict[str, Any]) -> str:
    scene = _clean_string(value)

    # Qwen2-VL-2B有时会保守输出“其他”。若OCR存在明确场景证据，
    # 应使用OCR兜底结果，而不是直接保留“其他”。
    if scene == "其他":
        inferred = _infer_scene_type(ocr_result)
        return inferred if inferred != "其他" else "其他"

    if scene in SCENE_TYPES:
        return scene
    if scene in SCENE_ALIASES:
        return SCENE_ALIASES[scene]

    for alias, standard in SCENE_ALIASES.items():
        if alias in scene:
            return standard

    return _infer_scene_type(ocr_result)


def _infer_scene_type(ocr_result: Dict[str, Any]) -> str:
    """Bilingual deterministic OCR fallback scene inference."""
    analysis = ocr_result.get("text_analysis")
    if not isinstance(analysis, dict):
        analysis = analyze_text_semantics(ocr_result)
    scores = dict(analysis.get("scene_keyword_scores", {}) or {})
    if analysis.get("price_text") or analysis.get("promotion_words"):
        scores["电商商品广告"] = scores.get("电商商品广告", 0.0) + 2.0
    if analysis.get("cta_text"):
        scores["电商商品广告"] = scores.get("电商商品广告", 0.0) + 0.6
    if analysis.get("brand_candidates"):
        scores["品牌广告"] = scores.get("品牌广告", 0.0) + 0.8
    if not scores:
        return "其他"
    scene, score = max(scores.items(), key=lambda item: item[1])
    return scene if score >= 1.5 else "其他"


def _extract_event_time(text: str) -> str:
    patterns = (
        r"(?:直播时间|活动时间|演出时间|时间)\s*[：:]\s*"
        r"([0-9]{4}[./\-年][0-9]{1,2}[./\-月][0-9]{1,2}日?"
        r"(?:\s*[0-9]{1,2}[：:][0-9]{2})?)",
        r"([0-9]{4}[./\-][0-9]{1,2}[./\-][0-9]{1,2}"
        r"\s*[0-9]{1,2}[：:][0-9]{2})",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return re.sub(r"\s+", " ", match.group(1)).strip()
    return ""


def _unwrap_nested_analysis(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    修复Qwen把完整分析对象错误塞进scene_evidence数组的情况。

    常见异常结构：
    {
      "scene_type": "其他",
      "scene_evidence": [{"main_subject": ..., "brand_info": ...}]
    }

    此函数会把嵌套对象中的标准字段恢复到顶层。
    """
    if not isinstance(data, dict):
        return {}

    scene_evidence = data.get("scene_evidence")
    if not isinstance(scene_evidence, list):
        return data

    nested_candidates = [item for item in scene_evidence if isinstance(item, dict)]
    if not nested_candidates:
        return data

    expected_keys = {
        "scene_type", "scene_subtype", "scene_evidence", "main_subject",
        "visible_evidence", "semantic_inference", "text_analysis",
        "product_type", "target_audience", "selling_points", "price_info",
        "promotion_info", "cta_info", "brand_info", "event_info",
        "emotion_style", "trust_signals", "attention_elements",
        "memory_points", "risk_points", "confidence",
    }

    best = max(
        nested_candidates,
        key=lambda item: len(expected_keys.intersection(item.keys())),
    )
    overlap = expected_keys.intersection(best.keys())
    if len(overlap) < 4:
        return data

    repaired = dict(data)
    for key in expected_keys:
        nested_value = best.get(key)
        top_value = repaired.get(key)

        top_missing = (
            top_value is None
            or top_value == ""
            or top_value == []
            or top_value == {}
            or (key == "scene_type" and top_value == "其他")
        )
        if key in best and top_missing:
            repaired[key] = nested_value

    # scene_evidence本身只允许字符串。嵌套对象已恢复后清空异常对象。
    repaired["scene_evidence"] = [
        item for item in scene_evidence if isinstance(item, str) and item.strip()
    ]
    repaired["_nested_analysis_recovered"] = True
    return repaired



def _adapt_compact_output(data: Dict[str, Any]) -> Dict[str, Any]:
    """把紧凑输出转换为现有validator兼容结构。"""
    if not isinstance(data, dict):
        return {}
    adapted = dict(data)
    if "main_emotion" in adapted and "emotion_style" not in adapted:
        adapted["emotion_style"] = {
            "main_emotion": adapted.get("main_emotion") or "其他",
            "evidence": [],
        }
    adapted.setdefault("scene_subtype", "")
    adapted.setdefault("scene_evidence", [])
    adapted.setdefault("target_audience", "")
    adapted.setdefault("semantic_inference", [])
    adapted.setdefault("trust_signals", [])
    return adapted



_PROTOCOL_KEY_ALIASES = {
    "SCENE": "scene_type",
    "场景": "scene_type",
    "SUBJECT": "main_subject",
    "主体": "main_subject",
    "PRODUCT": "product_type",
    "产品": "product_type",
    "EVIDENCE": "visible_evidence",
    "可见证据": "visible_evidence",
    "SELLING": "selling_points",
    "视觉卖点": "selling_points",
    "EMOTION": "main_emotion",
    "情绪": "main_emotion",
    "ATTENTION": "attention_elements",
    "注意": "attention_elements",
    "MEMORY": "memory_points",
    "记忆": "memory_points",
    "RISK": "risk_points",
    "风险": "risk_points",
}

_PROTOCOL_LIST_FIELDS = {
    "visible_evidence",
    "selling_points",
    "attention_elements",
    "memory_points",
    "risk_points",
}


def _split_protocol_items(value: str, max_items: int = 3) -> List[str]:
    text = str(value or "").strip()
    if not text or text.upper() in {"NONE", "NULL", "N/A", "无", "空"}:
        return []
    parts = re.split(r"\s*[|｜;；]\s*", text)
    result: List[str] = []
    for part in parts:
        item = _clean_string(part.strip(" -•\t"))
        if not item or item.upper() in {"NONE", "NULL", "N/A"}:
            continue
        if item not in result:
            result.append(item)
        if len(result) >= max_items:
            break
    return result


def _parse_line_protocol(text: str) -> Dict[str, Any]:
    """解析单行/多行紧凑协议，兼容 =、:、中文冒号及 || 分隔。"""
    cleaned = _strip_code_fences(text or "")
    cleaned = re.sub(r"\s*\n\s*", " || ", cleaned).strip()

    label_pattern = re.compile(
        r"(?:^|\|\|)\s*(SCENE|SUBJECT|PRODUCT|EVIDENCE|SELLING|EMOTION|ATTENTION|MEMORY|RISK|"
        r"场景|主体|产品|可见证据|视觉卖点|情绪|注意|记忆|风险)\s*[:：=]\s*",
        flags=re.I,
    )
    matches = list(label_pattern.finditer(cleaned))
    if not matches:
        return {}

    parsed: Dict[str, Any] = {}
    seen_keys: List[str] = []
    for index, match in enumerate(matches):
        raw_key = match.group(1).strip()
        key = _PROTOCOL_KEY_ALIASES.get(raw_key.upper()) or _PROTOCOL_KEY_ALIASES.get(raw_key)
        if not key:
            continue
        value_start = match.end()
        value_end = matches[index + 1].start() if index + 1 < len(matches) else len(cleaned)
        value = cleaned[value_start:value_end].strip(" |;；\t")
        if key in _PROTOCOL_LIST_FIELDS:
            parsed[key] = _split_protocol_items(value)
        else:
            parsed[key] = _clean_string(value)
        seen_keys.append(key)

    useful = bool(
        parsed.get("main_subject")
        or parsed.get("product_type")
        or parsed.get("visible_evidence")
        or parsed.get("selling_points")
    )
    if not useful:
        return {}

    parsed["_protocol_seen_fields"] = list(dict.fromkeys(seen_keys))
    parsed["_protocol_complete"] = len(set(seen_keys)) >= 9
    return _adapt_compact_output(parsed)


def _decode_json_string(token: str) -> str:
    try:
        return json.loads(token)
    except Exception:
        return token.strip('"').replace('\\"', '"').replace('\\n', ' ')


def _recover_string_field(text: str, field: str) -> str:
    pattern = rf'"{re.escape(field)}"\s*:\s*("(?:\\.|[^"\\])*")'
    match = re.search(pattern, text, flags=re.S)
    return _clean_string(_decode_json_string(match.group(1))) if match else ""


def _recover_array_field(text: str, field: str, max_items: int = 3) -> List[str]:
    """从截断JSON数组中恢复值；对象数组只取description/text/content/name等值，绝不把键名当证据。"""
    match = re.search(rf'"{re.escape(field)}"\s*:\s*\[', text)
    if not match:
        return []

    tail = text[match.end():]
    end = tail.find(']')
    if end >= 0:
        segment = tail[:end]
    else:
        next_field = re.search(
            r'\n\s*"(?:scene_type|main_subject|product_type|visible_evidence|selling_points|main_emotion|attention_elements|memory_points|risk_points)"\s*:',
            tail,
        )
        segment = tail[: next_field.start()] if next_field else tail

    preferred_values: List[str] = []
    for key in ("description", "text", "content", "value", "name"):
        pattern = rf'"{key}"\s*:\s*("(?:\\.|[^"\\])*")'
        for token in re.findall(pattern, segment, flags=re.S):
            value = _clean_string(_decode_json_string(token))
            if value and value not in preferred_values:
                preferred_values.append(value)
            if len(preferred_values) >= max_items:
                return preferred_values
    if preferred_values:
        return preferred_values[:max_items]

    ignored_tokens = {
        "type", "description", "text", "content", "value", "name", "url",
        "scene_type", "main_subject", "product_type", "visible_evidence",
        "selling_points", "main_emotion", "attention_elements", "memory_points",
        "risk_points",
    }
    values: List[str] = []
    for token in re.findall(r'"(?:\\.|[^"\\])*"', segment, flags=re.S):
        value = _clean_string(_decode_json_string(token))
        if not value or value in ignored_tokens:
            continue
        if value not in values:
            values.append(value)
        if len(values) >= max_items:
            break
    return values

def _recover_partial_compact_output(text: str) -> Dict[str, Any]:
    """解析失败时恢复前部完整字段，避免主体和视觉语义全部丢失。"""
    cleaned = _strip_code_fences(text or "")
    recovered: Dict[str, Any] = {
        "scene_type": _recover_string_field(cleaned, "scene_type"),
        "main_subject": _recover_string_field(cleaned, "main_subject"),
        "product_type": _recover_string_field(cleaned, "product_type"),
        "visible_evidence": _recover_array_field(cleaned, "visible_evidence"),
        "selling_points": _recover_array_field(cleaned, "selling_points"),
        "main_emotion": _recover_string_field(cleaned, "main_emotion"),
        "attention_elements": _recover_array_field(cleaned, "attention_elements"),
        "memory_points": _recover_array_field(cleaned, "memory_points"),
        "risk_points": _recover_array_field(cleaned, "risk_points"),
    }
    useful = bool(
        recovered["main_subject"]
        or recovered["product_type"]
        or recovered["visible_evidence"]
        or recovered["selling_points"]
    )
    return _adapt_compact_output(recovered) if useful else {}


def _sanitize_analysis(
    data: Dict[str, Any],
    ocr_result: Dict[str, Any],
) -> Dict[str, Any]:
    if not isinstance(data, dict):
        data = {}

    data = _adapt_compact_output(data)
    data = _unwrap_nested_analysis(data)

    cleaned: Dict[str, Any] = {
        "scene_type": _normalize_scene_type(data.get("scene_type"), ocr_result),
        "scene_subtype": _clean_string(data.get("scene_subtype")),
        "scene_evidence": _clean_list(data.get("scene_evidence"), max_items=4),
        "main_subject": _clean_string(data.get("main_subject")),
        "product_type": _clean_string(data.get("product_type")),
        "target_audience": _clean_string(data.get("target_audience")),
        "visible_evidence": _clean_list(data.get("visible_evidence"), max_items=6),
        "semantic_inference": _clean_list(data.get("semantic_inference"), max_items=4),
        "selling_points": _clean_list(data.get("selling_points"), max_items=6),
        "trust_signals": _clean_list(data.get("trust_signals"), max_items=6),
        "attention_elements": _clean_list(
            data.get("attention_elements"), max_items=5
        ),
        "memory_points": _clean_list(data.get("memory_points"), max_items=5),
        "risk_points": _clean_list(data.get("risk_points"), max_items=5),
    }

    if cleaned["product_type"] in {"商品", "产品", "内容"}:
        cleaned["product_type"] = ""

    price_info = (
        data.get("price_info")
        if isinstance(data.get("price_info"), dict)
        else {}
    )
    price_text = _clean_string(price_info.get("price_text"))
    cleaned["price_info"] = {
        "has_price": _to_bool(price_info.get("has_price")) and bool(price_text),
        "price_text": price_text,
    }

    promotion_info = (
        data.get("promotion_info")
        if isinstance(data.get("promotion_info"), dict)
        else {}
    )
    promotion_words = _clean_list(
        promotion_info.get("promotion_words"), max_items=5
    )
    cleaned["promotion_info"] = {
        "has_promotion": (
            _to_bool(promotion_info.get("has_promotion"))
            and bool(promotion_words)
        ),
        "promotion_words": promotion_words,
    }

    cta_info = (
        data.get("cta_info")
        if isinstance(data.get("cta_info"), dict)
        else {}
    )
    cta_text = _clean_string(cta_info.get("cta_text"))
    cleaned["cta_info"] = {
        "has_cta": _to_bool(cta_info.get("has_cta")) and bool(cta_text),
        "cta_text": cta_text,
    }

    brand_info = (
        data.get("brand_info")
        if isinstance(data.get("brand_info"), dict)
        else {}
    )
    brand_text = _clean_string(brand_info.get("brand_text"))
    cleaned["brand_info"] = {
        "has_brand": _to_bool(brand_info.get("has_brand")) and bool(brand_text),
        "brand_text": brand_text,
    }

    event_info = (
        data.get("event_info")
        if isinstance(data.get("event_info"), dict)
        else {}
    )
    event_name = _clean_string(event_info.get("event_name"))
    event_time = _clean_string(event_info.get("event_time"))
    event_location = _clean_string(event_info.get("event_location"))
    organizer = _clean_string(event_info.get("organizer"))
    has_event_fields = any(
        (event_name, event_time, event_location, organizer)
    )
    cleaned["event_info"] = {
        "has_event": (
            _to_bool(event_info.get("has_event"))
            or has_event_fields
            or cleaned["scene_type"] == "活动宣传海报"
        ),
        "event_name": event_name,
        "event_time": event_time,
        "event_location": event_location,
        "organizer": organizer,
    }

    emotion_style = (
        data.get("emotion_style")
        if isinstance(data.get("emotion_style"), dict)
        else {}
    )
    cleaned["emotion_style"] = {
        "main_emotion": (
            _clean_string(emotion_style.get("main_emotion")) or "其他"
        ),
        "evidence": _clean_list(
            emotion_style.get("evidence"), max_items=3
        ),
    }

    text_analysis_raw = (
        data.get("text_analysis")
        if isinstance(data.get("text_analysis"), dict)
        else {}
    )
    text_status = _clean_string(text_analysis_raw.get("status")).lower()
    if text_status not in {"absent", "readable", "partially_readable", "garbled"}:
        text_status = "readable" if (ocr_result.get("all_text") or []) else "absent"
    cleaned["text_analysis"] = {
        "status": text_status,
        "headline": _clean_string(text_analysis_raw.get("headline")),
        "brand_candidate": _clean_string(text_analysis_raw.get("brand_candidate")),
        "suspicious_text": _clean_list(text_analysis_raw.get("suspicious_text"), max_items=8),
    }

    confidence = (
        data.get("confidence")
        if isinstance(data.get("confidence"), dict)
        else {}
    )
    confidence_keys = (
        "scene_judgment",
        "main_subject",
        "ocr_understanding",
        "emotion_judgment",
        "overall",
    )
    clean_confidence = {
        key: _to_confidence(confidence.get(key, 0.0))
        for key in confidence_keys
    }

    # 仅当模型提供了分项置信度但漏填overall时，才计算分项平均值。
    component_values = [
        clean_confidence[key]
        for key in (
            "scene_judgment",
            "main_subject",
            "ocr_understanding",
            "emotion_judgment",
        )
        if clean_confidence[key] > 0
    ]
    confidence_source = "model"
    if clean_confidence["overall"] == 0.0 and component_values:
        clean_confidence["overall"] = round(
            sum(component_values) / len(component_values), 4
        )
        confidence_source = "derived_from_model_components"
    elif not component_values and clean_confidence["overall"] == 0.0:
        confidence_source = "unavailable"

    cleaned["confidence"] = clean_confidence
    cleaned["_confidence_source"] = confidence_source
    cleaned["_nested_analysis_recovered"] = bool(
        data.get("_nested_analysis_recovered")
    )
    cleaned["_template_echo_cleaned"] = True
    cleaned = _apply_ocr_fact_gates(cleaned, ocr_result)
    return cleaned


def _fallback_analysis(
    ocr_result: Dict[str, Any],
    visual_result: Dict[str, Any],
) -> Dict[str, Any]:
    joined = str(ocr_result.get("joined_text", "") or "")
    inferred_scene = _infer_scene_type(ocr_result)
    event_time = _extract_event_time(joined)
    return {
        "scene_type": inferred_scene,
        "scene_subtype": "",
        "scene_evidence": (
            ["根据OCR关键词进行兜底场景判断"] if joined else []
        ),
        "main_subject": "",
        "product_type": "",
        "target_audience": "",
        "visible_evidence": [],
        "semantic_inference": [],
        "selling_points": [],
        "price_info": {
            "has_price": bool(ocr_result.get("has_price")),
            "price_text": ",".join(ocr_result.get("price_text", [])),
        },
        "promotion_info": {
            "has_promotion": bool(ocr_result.get("has_promotion")),
            "promotion_words": ocr_result.get("promotion_words", []),
        },
        "cta_info": {
            "has_cta": bool(ocr_result.get("has_cta")),
            "cta_text": ",".join(ocr_result.get("cta_text", [])),
        },
        "brand_info": {"has_brand": False, "brand_text": ""},
        "event_info": {
            "has_event": (
                inferred_scene == "活动宣传海报" or bool(event_time)
            ),
            "event_name": "",
            "event_time": event_time,
            "event_location": "",
            "organizer": "",
        },
        "text_analysis": {
            "status": "readable" if (ocr_result.get("all_text") or []) else "absent",
            "headline": "",
            "brand_candidate": "",
            "suspicious_text": [],
        },
        "emotion_style": {"main_emotion": "其他", "evidence": []},
        "trust_signals": [],
        "attention_elements": [],
        "memory_points": [],
        "risk_points": [
            "Qwen模型未成功调用，当前仅使用OCR和基础视觉特征。"
        ],
        "confidence": {
            "scene_judgment": 0.0,
            "main_subject": 0.0,
            "ocr_understanding": 0.0,
            "emotion_judgment": 0.0,
            "overall": 0.0,
        },
        "_confidence_source": "unavailable",
        "fallback": True,
    }

def _compact_ocr(ocr_result: Dict[str, Any]) -> Dict[str, Any]:
    """仅提供规则化场景提示，不向Qwen传入OCR全文，避免模型抄写长文案。"""
    deterministic = ocr_result.get("text_analysis")
    if not isinstance(deterministic, dict):
        deterministic = analyze_text_semantics(ocr_result)

    scores = dict(deterministic.get("scene_keyword_scores", {}) or {})
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:3]
    return {
        "文字状态": deterministic.get("status", "absent"),
        "场景候选": [name for name, score in ranked if float(score or 0) > 0],
        "存在促销": bool(deterministic.get("promotion_words")),
        "存在CTA": bool(deterministic.get("cta_text")),
        "存在活动信息": bool(
            deterministic.get("event_dates")
            or deterministic.get("event_times")
            or deterministic.get("event_locations")
        ),
    }


def _compact_visual(visual_result: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "image_size",
        "brightness",
        "contrast",
        "saturation",
        "colorfulness",
        "sharpness",
        "edge_density",
        "border_complexity",
        "layout_complexity",
        "center_focus",
        "saliency_center",
        "dominant_colors",
        "quality_flags",
    ]
    return {key: visual_result.get(key) for key in keys if key in visual_result}


def _box_to_rect(box: Any, width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    """兼容[x1,y1,x2,y2]和四点框，返回裁剪后的矩形。"""
    if box is None:
        return None
    try:
        import numpy as np
        arr = np.asarray(box, dtype=float)
        if arr.ndim == 1 and arr.size >= 4:
            x1, y1, x2, y2 = arr[:4]
        else:
            arr = arr.reshape(-1, 2)
            x1, y1 = arr.min(axis=0)
            x2, y2 = arr.max(axis=0)
    except Exception:
        return None

    margin = max(2, int(round(min(width, height) * 0.004)))
    left = max(0, int(x1) - margin)
    top = max(0, int(y1) - margin)
    right = min(width - 1, int(x2) + margin)
    bottom = min(height - 1, int(y2) + margin)
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _create_text_masked_image(
    image_file: Path,
    ocr_result: Dict[str, Any],
) -> Tuple[Path, Optional[Path], int, float]:
    """为Qwen构造去文字视觉视图；OCR仍使用原图。"""
    if os.environ.get("QWEN_MASK_TEXT", "1").strip().lower() in {"0", "false", "no"}:
        return image_file, None, 0, 0.0

    items = ocr_result.get("text_items", []) or []
    if not items:
        return image_file, None, 0, 0.0

    try:
        import cv2
        import numpy as np

        image = cv2.imread(str(image_file), cv2.IMREAD_COLOR)
        if image is None:
            return image_file, None, 0, 0.0
        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        count = 0
        for item in items:
            if not isinstance(item, dict):
                continue
            rect = _box_to_rect(item.get("box"), width, height)
            if rect is None:
                continue
            left, top, right, bottom = rect
            cv2.rectangle(mask, (left, top), (right, bottom), 255, thickness=-1)
            count += 1

        if count == 0:
            return image_file, None, 0, 0.0

        ratio = float((mask > 0).mean())
        # 遮罩面积过大时避免严重修补畸变，改用柔和模糊。
        if ratio <= 0.35:
            masked = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
        else:
            blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=11, sigmaY=11)
            alpha = (mask.astype(np.float32) / 255.0)[..., None]
            masked = (image * (1.0 - alpha) + blurred * alpha).astype(np.uint8)

        handle = tempfile.NamedTemporaryFile(prefix="qwen_visual_", suffix=".png", delete=False)
        handle.close()
        temp_path = Path(handle.name)
        if not cv2.imwrite(str(temp_path), masked):
            temp_path.unlink(missing_ok=True)
            return image_file, None, 0, ratio
        return temp_path, temp_path, count, round(ratio, 6)
    except Exception:
        return image_file, None, 0, 0.0


def _build_messages(image_file: Path, prompt: str) -> List[Dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_file)},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def _generate_output(
    model: Any,
    processor: Any,
    image_file: Path,
    prompt: str,
    max_new_tokens: int,
    assistant_prefix: str = "SCENE=",
) -> Tuple[str, str]:
    import torch
    from qwen_vl_utils import process_vision_info

    messages = _build_messages(image_file, prompt)
    chat_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    # 预填充首字段，减少模型转为OCR抄录或自由描述的概率。
    chat_text = chat_text + assistant_prefix
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[chat_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    device = next(model.parameters()).device
    inputs = inputs.to(device)
    input_length = inputs["input_ids"].shape[1]

    generation_config = copy.deepcopy(model.generation_config)
    generation_config.do_sample = False
    generation_config.temperature = None
    generation_config.top_p = None
    generation_config.top_k = None
    generation_config.max_new_tokens = max_new_tokens
    generation_config.repetition_penalty = 1.12
    generation_config.use_cache = True

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            generation_config=generation_config,
        )

    continuation = processor.batch_decode(
        generated_ids[:, input_length:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    output = assistant_prefix + continuation
    return output.strip(), str(device)


def analyze_image_structured(
    image_path: str,
    ocr_result: Dict[str, Any],
    visual_result: Dict[str, Any],
    model_name: Optional[str] = None,
    use_qwen: bool = True,
) -> Dict[str, Any]:
    if not use_qwen:
        return _fallback_analysis(ocr_result, visual_result)

    try:
        import transformers

        image_file = Path(image_path).expanduser().resolve()
        if not image_file.exists():
            raise FileNotFoundError(f"找不到图片：{image_file}")

        model, processor, resolved_model_name = _get_model(model_name)
        prompt = STRUCTURED_PROMPT_TEMPLATE
        qwen_image, temporary_image, masked_box_count, masked_area_ratio = _create_text_masked_image(
            image_file, ocr_result
        )
        try:
            output, device = _generate_output(
                model=model,
                processor=processor,
                image_file=qwen_image,
                prompt=prompt,
                max_new_tokens=128,
                assistant_prefix="SCENE=",
            )
        finally:
            if temporary_image is not None:
                temporary_image.unlink(missing_ok=True)

        # v1.3.4优先使用助手前缀+紧凑协议，并让Qwen读取文字遮罩图。
        protocol_data = _parse_line_protocol(output)
        if protocol_data:
            data = _sanitize_analysis(protocol_data, ocr_result)
            data["_raw_model_output"] = output
            data["fallback"] = False
            data["_runtime"] = {
                "model_name": resolved_model_name,
                "transformers_version": transformers.__version__,
                "device": device,
                "json_retry": False,
                "second_multimodal_inference": False,
                "parse_error": False,
                "output_format": "line_protocol",
                "protocol_complete": bool(protocol_data.get("_protocol_complete")),
                "protocol_seen_fields": protocol_data.get("_protocol_seen_fields", []),
                "output_chars": len(output),
                "max_new_tokens": 128,
                "max_pixels": 512 * 28 * 28,
                "visual_input": "text_masked" if masked_box_count else "original",
                "masked_text_boxes": masked_box_count,
                "masked_area_ratio": masked_area_ratio,
            }
            return data

        # 若模型仍输出JSON，继续兼容；禁止第二次多模态推理。
        extracted = _extract_json(output)
        partial_recovery = False
        if extracted.get("parse_error"):
            recovered = _recover_partial_compact_output(output)
            if recovered:
                data = _sanitize_analysis(recovered, ocr_result)
                data["qwen_error"] = "OutputFormatError: 未遵守九行协议，已恢复可用视觉字段"
                data["fallback"] = False
                partial_recovery = True
            else:
                data = _fallback_analysis(ocr_result, visual_result)
                data["qwen_error"] = "OutputFormatError: 九行协议和JSON均无法解析，已启用本地兜底"
            data["_raw_model_output"] = output
            data["_runtime"] = {
                "model_name": resolved_model_name,
                "transformers_version": transformers.__version__,
                "device": device,
                "json_retry": False,
                "second_multimodal_inference": False,
                "parse_error": True,
                "partial_recovery": partial_recovery,
                "output_format": "recovered_or_fallback",
                "output_chars": len(output),
                "max_new_tokens": 128,
                "max_pixels": 512 * 28 * 28,
                "visual_input": "text_masked" if masked_box_count else "original",
                "masked_text_boxes": masked_box_count,
                "masked_area_ratio": masked_area_ratio,
            }
            return data

        data = _sanitize_analysis(_adapt_compact_output(extracted), ocr_result)
        data["_raw_model_output"] = output
        data["fallback"] = False
        data["_runtime"] = {
            "model_name": resolved_model_name,
            "transformers_version": transformers.__version__,
            "device": device,
            "json_retry": False,
            "second_multimodal_inference": False,
            "parse_error": False,
            "output_format": "json_compat",
            "output_chars": len(output),
            "max_new_tokens": 128,
            "max_pixels": 512 * 28 * 28,
        }
        return data

    except Exception as exc:
        data = _fallback_analysis(ocr_result, visual_result)
        data["qwen_error"] = f"{type(exc).__name__}: {exc}"
        return data


def analyze_image(
    image_path: str,
    objects: Optional[List[Dict[str, Any]]] = None,
    texts: Optional[List[str]] = None,
    sam_info: Optional[Dict[str, Any]] = None,
) -> str:
    ocr_result = {
        "all_text": texts or [],
        "joined_text": " ".join(texts or []),
    }
    visual_result = {
        "sam_info": sam_info or {},
        "legacy_yolo_objects": objects or [],
    }
    result = analyze_image_structured(
        image_path=image_path,
        ocr_result=ocr_result,
        visual_result=visual_result,
        use_qwen=True,
    )
    return json.dumps(result, ensure_ascii=False, indent=2)