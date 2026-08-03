"""Deterministic OCR text-role classification for advertisement images.

V1.3.7 separates visible text into roles and uses precision-first brand gating:
brand, title, slogan, CTA, promotion, price, event metadata and body text.
The classifier uses OCR text plus optional geometry. It does not query the web
and does not require a real-world brand dictionary, so AIGC-generated brand
marks can still be retained as visible evidence.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from modules.text_semantics import (
    clean_text,
    extract_ctas,
    extract_event_date_time,
    extract_locations,
    extract_prices,
    extract_promotions,
    is_watermark,
    normalize_text,
    unique_strings,
)


DIRECT_CTA_SCENE_PATTERNS: Mapping[str, Tuple[re.Pattern[str], ...]] = {
    "公益宣传": (
        re.compile(r"\bprotect\s+nature\b", re.I),
        re.compile(r"\breduce\s*(?:&|and)?\s*reuse\b", re.I),
        re.compile(r"\b(?:let'?s\s+)?build\s+a\s+better\s+tomorrow\b", re.I),
        re.compile(r"保护自然|保护环境|减少浪费|共建美好明天", re.I),
    ),
    "品牌广告": (
        re.compile(r"\bexperience\s+the\s+future\b", re.I),
        re.compile(r"\bexperience\s+ai\b", re.I),
        re.compile(r"体验未来|体验智能未来", re.I),
    ),
    "电商商品广告": (
        re.compile(r"\bexperience\s+the\s+future\b", re.I),
    ),
}

_GENERIC_BRAND_PHRASES = {
    "premium coffee", "premium skincare", "prestige skincare",
    "natural radiance", "urban mobility", "urbanmobility", "the future", "the future of",
    "innovation", "innovation in your hand", "innovation meets elegance",
    "innovate for life", "drive the future", "learn new skills",
    "grow your future", "learn anytime anywhere", "master any skill", "any skill", "anyskill", "demand skills", "demandskills",
    "make short videos", "videosd", "education", "tips tricks", "for beginners",
    "our earth", "our planet", "our responsibility", "sustainable future",
    "tropical", "paradise", "tropical island", "island paradise", "islandparadise", "awaits",
    "club", "university", "student union", "fitness", "forge fitness",
    "forge your", "best self", "for your everyday", "foryour everyday", "new member deal", "coming soon",
    "smart innovation", "max efficiency", "premium experience",
    "the future of intelligence", "intelligence", "smart", "performance",
    "coffee", "skincare", "technology", "future", "scene", "d scene",
}

_GENERIC_BRAND_TOKENS = {
    "premium", "coffee", "skincare", "prestige", "natural", "radiance",
    "future", "innovation", "urban", "mobility", "learn", "learning",
    "skills", "grow", "your", "our", "the", "earth", "planet",
    "responsibility", "sustainable", "tropical", "island", "paradise",
    "club", "university", "fitness", "forge", "best", "self", "scene",
    "smart", "performance", "experience", "efficiency", "education",
    "master", "make", "short", "videos", "technology", "intelligence", "secret", "trick", "beginner", "beginners", "any", "skill", "awaits", "demand",
    "drive", "innovate", "life", "member", "deal", "new", "coming",
    "soon", "for", "quality", "every", "day", "style", "design", "energy", "battery", "display",
    "camera", "traction", "outsole", "upper", "responsive", "cushioning",
    "breathable", "supportive", "unleash", "fit", "bold", "ready",
}

_BRAND_REJECT_PATTERNS = (
    re.compile(r"^(?:shop|buy|join|learn|enroll|get|start|book|scan|sign|come|experience|protect|reduce|build|discover|explore|drive|innovate|make|master)\b", re.I),
    re.compile(r"^(?:thefuture|urbanmobility|islandparadise|demandskills|ourplanet|ourearth|learnnew|learnany|growyour|forgeyour|foryour|bestself|unleashyour|make|master|tropical|paradise)", re.I),
    re.compile(r"\b(?:off|free|trial|deal|offer|sale|discount|price|date|time|location|today|tomorrow)\b", re.I),
    re.compile(r"\b(?:system|battery|display|camera|connectivity|training|course|programs?|curriculum|certificate)\b", re.I),
    re.compile(r"\d"),
)


# Generic headings, topic labels, locations and campaign phrases must not be
# promoted to confirmed brands merely because they are large uppercase text.
_HARD_GENERIC_BRAND_WORDS = {
    "power", "cultural", "city", "trending", "trend", "alert", "hot",
    "hackathon", "challenge", "campus", "health", "awareness", "research",
    "innovation", "concept", "beyond", "limits", "exam", "success",
    "courses", "prep", "future", "technology", "performance", "sound",
    "freedom", "fresh", "watch", "must", "click", "available",
    "reliable", "something", "welcome", "impact", "progress", "ideas",
    "regular", "expert", "proven", "focus", "english", "trusted", "great",
}

_HARD_GENERIC_COMPACTS = {
    "performan", "performance", "perfeotion", "theartof", "startshere",
    "prepcourse", "mathematics", "science", "regular", "impact",
    "researchteam", "pioneeringsolutions", "innovationlab", "fresh",
    "industry", "learners", "hours", "create", "unlimited",
}

_COMMON_TITLE_WORDS = {
    "a", "ai", "the", "of", "for", "and", "your", "our", "pure", "find",
    "sound", "freedom", "power", "cultural", "city", "where", "history",
    "trend", "alert", "trending", "hot", "must", "watch", "click", "now",
    "hackathon", "challenge", "campus", "exam", "success", "start", "starts",
    "here", "ace", "health", "awareness", "run", "beyond", "limits",
    "research", "innovation", "concept", "future", "technology", "project",
    "name", "course", "courses", "prep", "performance", "fresh",
    "urban", "edge", "pulse", "fit", "neo", "velocity",
}

_TOPIC_PREFIXES = (
    "exam", "aceyour", "yourhealth", "ourhealth", "runbeyond", "research",
    "innovation", "trend", "campus", "hackathon", "health", "pure", "findyour",
)

_BRANDLIKE_AFFIXES = (
    "neo", "nex", "nova", "aure", "velo", "pulse", "urban", "eclat",
    "trop", "vyb", "wald", "nov", "fit", "edge", "pop", "ora", "ion",
)

_SCENE_GENERIC_TERMS = {
    "活动宣传海报": {"hackathon", "challenge", "campus", "competition", "event"},
    "教育校园宣传": {"exam", "success", "course", "courses", "prep", "campus"},
    "短视频封面": {"trendalert", "trending", "hot", "mustwatch", "clicknow"},
    "公益宣传": {"health", "awareness", "yourhealth", "ourconcern", "prevention"},
    "科技创新宣传": {"research", "innovation", "innovationconcept", "researchinnovation"},
}


def _word_tokens(text: str) -> List[str]:
    return [token.lower() for token in re.findall(r"[A-Za-z]+", clean_text(text))]


def _brand_language_assessment(
    text: str,
    *,
    scene_type: str,
    lines: Sequence[str],
    possible_brand_words: Sequence[str],
    score: float,
    geometry: Optional[Mapping[str, float]] = None,
) -> Dict[str, Any]:
    """Classify a geometric candidate as confirmed/candidate/rejected.

    The decision is intentionally precision-first. Ambiguous titles remain
    candidates for audit, but only confirmed candidates enter brand_info and
    downstream trust/memory scoring.
    """
    value = clean_text(text).strip(" |•·:;,.!?>→")
    compact = normalize_text(value)
    words = _word_tokens(value)
    positives: List[str] = []
    negatives: List[str] = []

    if not compact or len(compact) < 3:
        return {"status": "rejected", "positive_evidence": [], "negative_evidence": ["文本过短"]}

    possible_norms = {normalize_text(item) for item in possible_brand_words}
    if compact in possible_norms:
        positives.append("OCR品牌候选列表支持")

    repeats = sum(normalize_text(line) == compact for line in lines)
    if repeats >= 2:
        positives.append("同一标记重复出现")

    if any(char in value for char in ("'", "’", "É", "é", "À", "à")):
        positives.append("具有名称式拼写特征")

    if any(compact.startswith(prefix) or compact.endswith(prefix) for prefix in _BRANDLIKE_AFFIXES):
        positives.append("具有品牌式词形")

    # Hard semantic exclusions.
    area_ratio = float((geometry or {}).get("area_ratio", 0.0) or 0.0)
    height_ratio = float((geometry or {}).get("height_ratio", 0.0) or 0.0)
    if area_ratio < 0.0015 and height_ratio < 0.018 and not any(
        compact.startswith(prefix) or compact.endswith(prefix) for prefix in _BRANDLIKE_AFFIXES
    ):
        negatives.append("微小说明文字")
    if len(words) == 1 and words[0] in _HARD_GENERIC_BRAND_WORDS:
        negatives.append("单个通用主题词")
    if compact in _HARD_GENERIC_COMPACTS:
        negatives.append("通用标题或属性词")
    if compact.startswith(_TOPIC_PREFIXES):
        negatives.append("更像标题、主题或命令短语")
    if any(token in compact for token in ("mustwatch", "clicknow", "jointhechallenge", "registernow", "enrolltoday")):
        negatives.append("行动号召文本")
    if words and len(words) <= 5 and all(word in _COMMON_TITLE_WORDS for word in words):
        negatives.append("可由常见标题词完整组成")

    scene_terms = _SCENE_GENERIC_TERMS.get(scene_type, set())
    if compact in scene_terms:
        negatives.append(f"属于{scene_type}的通用场景词")

    # In tourism posters a prominent city name followed by WHERE/EXPLORE is a
    # destination title rather than a brand. This is structural, not a city list.
    joined_lower = " ".join(lines).lower()
    if scene_type == "旅游宣传" and ("where " in joined_lower or "explore" in joined_lower):
        if len(words) == 1 and score >= 6.0:
            negatives.append("旅游目的地标题")

    # A compact coined token or uncommon short name is positive evidence.
    if len(words) == 1 and 4 <= len(compact) <= 18:
        if words[0] not in _HARD_GENERIC_BRAND_WORDS and not compact.startswith(_TOPIC_PREFIXES):
            positives.append("短且独立的名称式词形")
    elif 1 <= len(words) <= 3 and not all(word in _COMMON_TITLE_WORDS for word in words):
        positives.append("短名称短语且非完整广告句")

    hard_negative = any(
        reason in {"单个通用主题词", "通用标题或属性词", "行动号召文本", "更像标题、主题或命令短语", "旅游目的地标题", "微小说明文字"}
        or reason.startswith("属于")
        for reason in negatives
    )
    # Generic theme words can never be confirmed without strong independent
    # evidence such as repetition or distinctive morphology.
    if negatives and not positives:
        status = "rejected"
    elif hard_negative:
        status = "rejected"
    elif negatives and not ("同一标记重复出现" in positives or "具有品牌式词形" in positives or "具有名称式拼写特征" in positives):
        status = "candidate"
    elif score >= 5.6 and len(positives) >= 2:
        status = "confirmed"
    elif score >= 6.5 and positives:
        status = "confirmed"
    elif score >= 4.8 and positives:
        status = "candidate"
    else:
        status = "rejected"

    return {
        "status": status,
        "positive_evidence": positives,
        "negative_evidence": negatives,
    }


def _safe_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [clean_text(item) for item in value if clean_text(item)]
    if value is None:
        return []
    text = clean_text(value)
    return [text] if text else []


def _infer_canvas(
    items: Sequence[Dict[str, Any]],
    image_size: Optional[Mapping[str, Any]],
) -> Tuple[float, float]:
    if image_size:
        try:
            width = float(image_size.get("width", 0) or 0)
            height = float(image_size.get("height", 0) or 0)
            if width > 0 and height > 0:
                return width, height
        except Exception:
            pass
    xs: List[float] = []
    ys: List[float] = []
    for item in items:
        box = item.get("box")
        if not isinstance(box, list):
            continue
        for point in box:
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                try:
                    xs.append(float(point[0]))
                    ys.append(float(point[1]))
                except Exception:
                    continue
    return (max(xs, default=1.0) * 1.03, max(ys, default=1.0) * 1.03)


def _geometry(item: Dict[str, Any], width: float, height: float) -> Dict[str, float]:
    box = item.get("box")
    if not isinstance(box, list) or not box:
        return {
            "x_ratio": 0.0, "y_ratio": 0.5, "width_ratio": 0.0,
            "height_ratio": 0.0, "area_ratio": 0.0,
        }
    points: List[Tuple[float, float]] = []
    for point in box:
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            try:
                points.append((float(point[0]), float(point[1])))
            except Exception:
                pass
    if not points:
        return {
            "x_ratio": 0.0, "y_ratio": 0.5, "width_ratio": 0.0,
            "height_ratio": 0.0, "area_ratio": 0.0,
        }
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    w = max(x1 - x0, 0.0)
    h = max(y1 - y0, 0.0)
    return {
        "x_ratio": round(x0 / max(width, 1.0), 4),
        "y_ratio": round(y0 / max(height, 1.0), 4),
        "width_ratio": round(w / max(width, 1.0), 4),
        "height_ratio": round(h / max(height, 1.0), 4),
        "area_ratio": round((w * h) / max(width * height, 1.0), 5),
    }


def _line_matches_fact(line: str, facts: Sequence[str]) -> bool:
    norm = normalize_text(line)
    if not norm:
        return False
    for fact in facts:
        fn = normalize_text(fact)
        if fn and (fn in norm or norm in fn):
            return True
    return False


def _canonical_cta(value: str) -> str:
    """Return a stable CTA category without overwriting visible OCR wording."""
    text = clean_text(value).strip(" |•·:;,.!?>→")
    norm = normalize_text(text)
    if norm.startswith("letsbuildabettertomorrow"):
        return "BUILD_A_BETTER_TOMORROW"
    mapping = {
        "buildabettertomorrow": "BUILD_A_BETTER_TOMORROW",
        "protectnature": "PROTECT_NATURE",
        "reducereuse": "REDUCE_REUSE",
        "learnmorex": "LEARN_MORE",
        "learnmore": "LEARN_MORE",
        "comejoinus": "JOIN_US",
        "joinus": "JOIN_US",
        "joinnow": "JOIN_NOW",
        "shopnow": "SHOP_NOW",
        "buynow": "BUY_NOW",
        "enrollnow": "ENROLL_NOW",
        "enrolltoday": "ENROLL_TODAY",
        "getstartednow": "GET_STARTED",
        "getstarted": "GET_STARTED",
        "startlearningtoday": "START_LEARNING",
        "startlearning": "START_LEARNING",
        "scanqr": "SCAN_QR",
        "scantheqr": "SCAN_QR",
        "signup": "SIGN_UP",
        "grabyoursnow": "GRAB_YOURS_NOW",
        "mustwatch": "MUST_WATCH",
        "clicknow": "CLICK_NOW",
        "registernow": "REGISTER_NOW",
        "jointhechallenge": "JOIN_THE_CHALLENGE",
        "startyoursuccessstory": "START_YOUR_SUCCESS_STORY",
        "experiencethefuture": "EXPERIENCE_THE_FUTURE",
        "booknow": "BOOK_NOW",
    }
    return mapping.get(norm, re.sub(r"[^A-Za-z0-9\u4e00-\u9fff]+", "_", text.upper()).strip("_"))


_CTA_SIGNATURES: Mapping[str, Tuple[str, ...]] = {
    "SHOP_NOW": ("shopnow",),
    "BUY_NOW": ("buynow",),
    "GRAB_YOURS_NOW": ("grabyoursnow",),
    "JOIN_NOW": ("joinnow",),
    "JOIN_US": ("joinus", "comejoinus"),
    "JOIN_THE_CHALLENGE": ("jointhechallenge",),
    "REGISTER_NOW": ("registernow",),
    "LEARN_MORE": ("learnmore", "learnmorex"),
    "ENROLL_NOW": ("enrollnow",),
    "ENROLL_TODAY": ("enrolltoday",),
    "GET_STARTED": ("getstarted", "getstartednow"),
    "START_LEARNING": ("startlearning", "startlearningtoday"),
    "START_YOUR_SUCCESS_STORY": ("startyoursuccessstory",),
    "MUST_WATCH": ("mustwatch",),
    "CLICK_NOW": ("clicknow",),
    "BOOK_NOW": ("booknow", "booktoday", "bookyour"),
    "SIGN_UP": ("signup",),
    "SCAN_QR": ("scanqr", "scantheqr"),
    "EXPERIENCE_THE_FUTURE": ("experiencethefuture",),
    "PROTECT_NATURE": ("protectnature",),
    "REDUCE_REUSE": ("reducereuse",),
    "BUILD_A_BETTER_TOMORROW": ("buildabettertomorrow", "letsbuildabettertomorrow"),
    "立即购买": ("立即购买", "马上购买", "点击购买", "立即抢购", "马上抢", "下单"),
    "立即报名": ("立即报名", "马上报名", "扫码报名", "报名参加"),
    "了解更多": ("了解更多", "点击查看", "点击观看"),
    "预约": ("立即预约", "预约咨询", "预约直播"),
}

_CTA_ACTION_PREFIXES = (
    "shop", "buy", "grab", "join", "comejoin", "register", "learn", "enroll",
    "getstarted", "startlearning", "startyour", "mustwatch", "click", "book",
    "signup", "scan", "experience", "protect", "reduce", "build", "letsbuild",
    "立即", "马上", "点击", "扫码", "报名", "预约",
)


def _box_extent(item: Mapping[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    box = item.get("box")
    if not isinstance(box, (list, tuple)) or len(box) < 2:
        return None
    try:
        xs = [float(point[0]) for point in box]
        ys = [float(point[1]) for point in box]
    except Exception:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def _cta_items_are_adjacent(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    canvas_width: float,
    canvas_height: float,
) -> bool:
    """Conservatively decide whether two OCR fragments may form one CTA."""
    a = _box_extent(left)
    b = _box_extent(right)
    if a is None or b is None:
        return True
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ah = max(ay1 - ay0, 1.0)
    bh = max(by1 - by0, 1.0)
    acy = (ay0 + ay1) / 2.0
    bcy = (by0 + by1) / 2.0
    acx = (ax0 + ax1) / 2.0
    bcx = (bx0 + bx1) / 2.0
    horizontal_same_line = (
        abs(acy - bcy) <= 0.8 * max(ah, bh)
        and max(bx0 - ax1, ax0 - bx1, 0.0) <= 0.12 * max(canvas_width, 1.0)
    )
    vertical_next_line = (
        0 <= by0 - ay1 <= 1.8 * max(ah, bh)
        and abs(acx - bcx) <= 0.28 * max(canvas_width, 1.0)
    )
    return horizontal_same_line or vertical_next_line


def _cta_candidate_windows(
    items: Sequence[Dict[str, Any]],
    width: float,
    height: float,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    usable = [(idx, item, clean_text(item.get("text"))) for idx, item in enumerate(items)]
    usable = [(idx, item, value) for idx, item, value in usable if value and not is_watermark(value)]
    for idx, item, value in usable:
        candidates.append({"raw": value, "indices": [idx], "span": 1})
    for pos in range(len(usable) - 1):
        idx1, item1, value1 = usable[pos]
        idx2, item2, value2 = usable[pos + 1]
        if idx2 != idx1 + 1:
            continue
        if _cta_items_are_adjacent(item1, item2, width, height):
            candidates.append({
                "raw": clean_text(value1 + " " + value2),
                "indices": [idx1, idx2],
                "span": 2,
            })
    return candidates


def _candidate_matches_cta(canonical: str, raw: str) -> bool:
    norm = normalize_text(raw)
    if not norm:
        return False
    signatures = _CTA_SIGNATURES.get(canonical, ())
    if canonical == "BOOK_NOW":
        return norm.startswith("book") and any(sig in norm for sig in signatures)
    if canonical == "BUILD_A_BETTER_TOMORROW":
        return "buildabettertomorrow" in norm
    return any(signature in norm for signature in signatures)


def _repair_cta_display(raw: str, canonical: str) -> str:
    """Repair spacing only; never add semantic words absent from OCR evidence."""
    value = clean_text(raw).strip(" |•·:;,.?>→")
    ending = "!" if clean_text(raw).rstrip().endswith("!") else ""
    norm = normalize_text(value)
    exact = {
        "comejoinus": "COME JOIN US",
        "joinus": "JOIN US",
        "joinnow": "JOIN NOW",
        "shopnow": "SHOP NOW",
        "buynow": "BUY NOW",
        "enrollnow": "ENROLL NOW",
        "enrolltoday": "ENROLL TODAY",
        "getstartednow": "GET STARTED NOW",
        "getstarted": "GET STARTED",
        "startlearningtoday": "START LEARNING TODAY",
        "startlearning": "START LEARNING",
        "grabyoursnow": "GRAB YOURS NOW",
        "mustwatch": "MUST WATCH",
        "clicknow": "CLICK NOW",
        "registernow": "REGISTER NOW",
        "jointhechallenge": "JOIN THE CHALLENGE",
        "startyoursuccessstory": "START YOUR SUCCESS STORY",
        "learnmore": "LEARN MORE",
        "learnmorex": "LEARN MORE",
        "letsbuildabettertomorrowtogether": "LET'S BUILD A BETTER TOMORROW, TOGETHER",
        "buildabettertomorrowtogether": "BUILD A BETTER TOMORROW, TOGETHER",
        "buildabettertomorrow": "BUILD A BETTER TOMORROW",
        "protectnature": "PROTECT NATURE",
        "reducereuse": "REDUCE & REUSE",
        "experiencethefuture": "EXPERIENCE THE FUTURE",
    }
    if norm in exact:
        repaired = exact[norm]
    else:
        repaired = value
        # Common OCR joins at terminal CTA words. These insert spaces only.
        repaired = re.sub(r"(?<=[A-Za-z])(?=TODAY\b)", " ", repaired, flags=re.I)
        repaired = re.sub(r"(?<=[A-Za-z])(?=NOW\b)", " ", repaired, flags=re.I)
        repaired = re.sub(r"^(COME)(?=JOIN\b)", r"\1 ", repaired, flags=re.I)
        repaired = re.sub(r"^(START)(?=LEARNING\b)", r"\1 ", repaired, flags=re.I)
        repaired = re.sub(r"^(GET)(?=STARTED\b)", r"\1 ", repaired, flags=re.I)
        repaired = re.sub(r"^(LET'?S)(?=BUILD\b)", r"\1 ", repaired, flags=re.I)
        repaired = re.sub(r"\s+", " ", repaired).strip()
    if ending and not repaired.endswith("!"):
        repaired += "!"
    return repaired


_CTA_CANONICAL_DISPLAY: Mapping[str, str] = {
    "SHOP_NOW": "SHOP NOW",
    "BUY_NOW": "BUY NOW",
    "GRAB_YOURS_NOW": "GRAB YOURS NOW",
    "JOIN_NOW": "JOIN NOW",
    "JOIN_US": "JOIN US",
    "JOIN_THE_CHALLENGE": "JOIN THE CHALLENGE",
    "REGISTER_NOW": "REGISTER NOW",
    "LEARN_MORE": "LEARN MORE",
    "ENROLL_NOW": "ENROLL NOW",
    "ENROLL_TODAY": "ENROLL TODAY",
    "GET_STARTED": "GET STARTED",
    "START_LEARNING": "START LEARNING",
    "START_YOUR_SUCCESS_STORY": "START YOUR SUCCESS STORY",
    "MUST_WATCH": "MUST WATCH",
    "CLICK_NOW": "CLICK NOW",
    "BOOK_NOW": "BOOK NOW",
    "SIGN_UP": "SIGN UP",
    "SCAN_QR": "SCAN THE QR",
    "SCAN_QR_SIGN_UP": "SCAN THE QR TO SIGN UP",
    "EXPERIENCE_THE_FUTURE": "EXPERIENCE THE FUTURE",
    "PROTECT_NATURE": "PROTECT NATURE",
    "REDUCE_REUSE": "REDUCE & REUSE",
    "BUILD_A_BETTER_TOMORROW": "BUILD A BETTER TOMORROW",
}


def _extract_cta_evidence(
    canonical_values: Sequence[str],
    items: Sequence[Dict[str, Any]],
    width: float,
    height: float,
) -> Tuple[List[str], List[str], List[Dict[str, Any]], List[str]]:
    """Resolve canonical CTA categories back to complete OCR-supported wording."""
    canonical_values = unique_strings(canonical_values)
    candidates = _cta_candidate_windows(items, width, height)
    evidence: List[Dict[str, Any]] = []
    seen_pairs = set()

    for canonical in canonical_values:
        matches: List[Tuple[float, Dict[str, Any]]] = []
        for candidate in candidates:
            raw = candidate["raw"]
            if not _candidate_matches_cta(canonical, raw):
                continue
            norm = normalize_text(raw)
            signatures = _CTA_SIGNATURES.get(canonical, ())
            starts_this_cta = any(norm.startswith(signature) for signature in signatures)
            if canonical == "BOOK_NOW":
                starts_this_cta = norm.startswith("book")
            elif canonical == "BUILD_A_BETTER_TOMORROW":
                starts_this_cta = norm.startswith(("buildabettertomorrow", "letsbuildabettertomorrow"))
            score = 6.0 if starts_this_cta else 1.0
            score += min(len(norm), 80) / 120.0
            if candidate.get("span") == 1:
                score += 1.5
            else:
                second_norm = normalize_text(raw.split(" ")[-1])
                if second_norm in {"now", "today", "us", "more", "signup", "challenge", "together"}:
                    score += 1.0
                else:
                    score -= 0.6
            if any(token in norm for token in ("today", "now", "together")):
                score += 0.4
            matches.append((score, candidate))

        if not matches:
            continue
        matches.sort(key=lambda pair: (-pair[0], pair[1].get("span", 1), pair[1]["indices"][0]))
        best = matches[0][1]
        raw = clean_text(best["raw"])
        raw_norm = normalize_text(raw)
        pair_key = (raw_norm, canonical)
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)

        # One OCR line may contain two CTA commands without spaces, e.g.
        # JOIN NOWSTARTYOURSUCCESSSTORY. In this case each canonical field gets
        # its own OCR-supported readable segment instead of duplicating the
        # entire combined line.
        matched_canonicals = [
            value for value in canonical_values
            if _candidate_matches_cta(value, raw)
        ]
        if len(matched_canonicals) >= 2 and canonical not in {"SCAN_QR", "SIGN_UP"}:
            display = _CTA_CANONICAL_DISPLAY.get(canonical, _repair_cta_display(raw, canonical))
        else:
            display = _repair_cta_display(raw, canonical)

        evidence.append({
            "cta_text": display,
            "cta_raw_text": raw,
            "cta_canonical": canonical,
            "ocr_indices": list(best.get("indices", [])),
            "source": "ocr_text_role_rules",
        })

    # Merge adjacent "Scan the QR" + "to sign up" fragments into one complete
    # OCR-supported CTA rather than exposing two partial phrases.
    scan = next((item for item in evidence if item["cta_canonical"] == "SCAN_QR"), None)
    signup = next((item for item in evidence if item["cta_canonical"] == "SIGN_UP"), None)
    if scan and signup:
        scan_indices = scan.get("ocr_indices", [])
        signup_indices = signup.get("ocr_indices", [])
        adjacent = bool(scan_indices and signup_indices and abs(min(scan_indices) - min(signup_indices)) <= 1)
        if adjacent:
            ordered = sorted([scan, signup], key=lambda item: min(item.get("ocr_indices", [9999])))
            raw = clean_text(" ".join(item["cta_raw_text"] for item in ordered))
            combined = {
                "cta_text": "SCAN THE QR TO SIGN UP",
                "cta_raw_text": raw,
                "cta_canonical": "SCAN_QR_SIGN_UP",
                "ocr_indices": sorted(set(scan_indices + signup_indices)),
                "source": "ocr_adjacent_text_role_rules",
            }
            first_position = min(evidence.index(scan), evidence.index(signup))
            evidence = [item for item in evidence if item is not scan and item is not signup]
            evidence.insert(first_position, combined)

    display_values = unique_strings(item["cta_text"] for item in evidence)
    raw_values = unique_strings(item["cta_raw_text"] for item in evidence)
    resolved_canonical = unique_strings(item["cta_canonical"] for item in evidence)
    return display_values, raw_values, evidence, resolved_canonical

def _contextual_ctas(lines: Sequence[str], scene_type: str) -> List[str]:
    found: List[str] = []
    # Regex matching may use adjacent-line windows for split OCR.
    windows = list(lines)
    for index in range(len(lines) - 1):
        windows.append(clean_text(lines[index] + " " + lines[index + 1]))
    patterns = DIRECT_CTA_SCENE_PATTERNS.get(scene_type, ())
    for candidate in windows:
        for pattern in patterns:
            match = pattern.search(candidate)
            if match:
                found.append(clean_text(match.group(0)))

    # Compact OCR repair is restricted to individual lines to avoid combining
    # unrelated neighboring slogans into one CTA.
    for candidate in lines:
        compact = normalize_text(candidate)
        if scene_type == "公益宣传":
            if "protectnature" in compact:
                found.append("PROTECT NATURE")
            if "reducereuse" in compact:
                found.append("REDUCE & REUSE")
            if "buildabettertomorrow" in compact:
                found.append(clean_text(candidate))
        if scene_type in {"品牌广告", "电商商品广告"} and "experiencethefuture" in compact:
            found.append("EXPERIENCE THE FUTURE")

    canonical = unique_strings(_canonical_cta(value) for value in found)
    return canonical


def _brand_display(candidate: str, all_lines: Sequence[str]) -> str:
    value = clean_text(candidate).strip(" |•·:;,.!?>→")
    compact = re.sub(r"[^A-Za-z0-9]", "", value).upper()
    tokens = [
        re.sub(r"[^A-Za-z0-9]", "", clean_text(line)).upper()
        for line in all_lines
    ]
    tokens = [token for token in tokens if 2 <= len(token) <= 14]

    # Reconstruct compact two-part marks when both parts are visible elsewhere.
    best_pair: Optional[Tuple[str, str]] = None
    for left in tokens:
        for right in tokens:
            if left == right:
                continue
            if len(left) < 3 or len(right) < 2:
                continue
            if left + right == compact:
                best_pair = (left, right)
                break
        if best_pair:
            break
    if best_pair:
        return f"{best_pair[0]} {best_pair[1]}"

    # Common compact AI suffix and single-letter model suffix formatting.
    if re.fullmatch(r"XN[A-Z]{3,10}AI", compact):
        compact = compact[1:]
    if re.fullmatch(r"[A-Z]{4,12}AI", compact) and len(compact) >= 6:
        return f"{compact[:-2]} AI"
    if re.fullmatch(r"[A-Z]{5,12}X", compact):
        return f"{compact[:-1]} X"
    return value


def _is_generic_brand(text: str) -> bool:
    value = clean_text(text).strip(" |•·:;,.!?>→")
    norm_space = re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
    compact = normalize_text(value)
    if not compact or len(compact) < 3 or len(value) > 28:
        return True
    if norm_space in _GENERIC_BRAND_PHRASES:
        return True
    if any(pattern.search(norm_space) for pattern in _BRAND_REJECT_PATTERNS):
        return True
    if any(pattern.search(compact) for pattern in _BRAND_REJECT_PATTERNS[:2]):
        return True
    if re.fullmatch(r"\d+(?:\.\d+)?", value):
        return True
    if any(char in value for char in "?$%\"<>[]{}"):
        return True
    words = re.findall(r"[A-Za-z]+", norm_space)
    if len(words) > 4:
        return True
    if words and all(word in _GENERIC_BRAND_TOKENS for word in words):
        return True
    if "scene" in words or "club" in words or "university" in words:
        return True
    return False


def _brand_score(
    text: str,
    index: int,
    geom: Mapping[str, float],
    lines: Sequence[str],
    possible_brand_words: Sequence[str],
) -> float:
    if _is_generic_brand(text):
        return -100.0
    value = clean_text(text).strip(" |•·:;,.!?>→")
    compact = re.sub(r"[^A-Za-z0-9]", "", value)
    if len(compact) < 3:
        return -100.0

    score = 0.0
    if index == 0:
        score += 2.0
    elif index <= 2:
        score += 0.6
    y = float(geom.get("y_ratio", 0.5) or 0.5)
    area = float(geom.get("area_ratio", 0.0) or 0.0)
    h = float(geom.get("height_ratio", 0.0) or 0.0)
    if y <= 0.12:
        score += 2.2
    elif y <= 0.22:
        score += 1.2
    if area >= 0.02:
        score += 2.2
    elif area >= 0.008:
        score += 1.4
    elif area >= 0.003:
        score += 0.5
    if h >= 0.04:
        score += 1.5
    elif h >= 0.022:
        score += 0.8

    if re.fullmatch(r"[A-Z][A-Z0-9&.' -]{2,20}", value):
        score += 1.8
    elif re.fullmatch(r"[A-Za-z][A-Za-z0-9&.' -]{2,20}", value):
        score += 1.0
    if value.isupper() and 4 <= len(compact) <= 14:
        score += 1.0
    if " " not in value and value.isupper() and 5 <= len(compact) <= 12:
        score += 0.8

    norm = normalize_text(value)
    repeats = sum(normalize_text(line) == norm for line in lines)
    if repeats >= 2:
        score += 1.5
    possible_norms = {normalize_text(item) for item in possible_brand_words}
    if norm in possible_norms:
        score += 1.0
    return round(score, 3)


def classify_text_roles(
    ocr_result: Dict[str, Any],
    scene_type: str = "",
    image_size: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    raw_items = ocr_result.get("text_items") or []
    items: List[Dict[str, Any]] = [item for item in raw_items if isinstance(item, dict)]
    if not items:
        items = [
            {"text": text, "confidence": None, "box": None}
            for text in _safe_list(ocr_result.get("all_text"))
        ]
    width, height = _infer_canvas(items, image_size)
    lines = [clean_text(item.get("text")) for item in items if clean_text(item.get("text"))]
    lines_no_watermark = [line for line in lines if not is_watermark(line)]

    prices = extract_prices(lines_no_watermark)
    promotions = extract_promotions(lines_no_watermark)
    direct_canonical = unique_strings(_canonical_cta(value) for value in extract_ctas(lines_no_watermark))
    contextual_canonical = _contextual_ctas(lines_no_watermark, scene_type)
    cta_canonical = unique_strings(direct_canonical + contextual_canonical)
    if "JOIN_NOW" in cta_canonical and "JOIN_US" in cta_canonical:
        cta_canonical = [value for value in cta_canonical if value != "JOIN_US"]
    ctas, cta_raw_text, cta_evidence, cta_canonical = _extract_cta_evidence(
        cta_canonical, items, width, height
    )
    direct_ctas, direct_raw_text, direct_evidence, direct_canonical = _extract_cta_evidence(
        direct_canonical, items, width, height
    )
    contextual_ctas, contextual_raw_text, contextual_evidence, contextual_canonical = _extract_cta_evidence(
        contextual_canonical, items, width, height
    )
    event_dt = extract_event_date_time(lines_no_watermark)
    locations = extract_locations(lines_no_watermark)
    possible_brand_words = _safe_list(ocr_result.get("possible_brand_words"))

    records: List[Dict[str, Any]] = []
    brand_scored: List[Tuple[int, float, str, int, Dict[str, Any]]] = []
    for index, item in enumerate(items):
        text = clean_text(item.get("text"))
        if not text:
            continue
        geom = _geometry(item, width, height)
        role = "unknown"
        reason = ""
        if is_watermark(text):
            role, reason = "watermark", "AI生成水印"
        elif _line_matches_fact(text, prices):
            role, reason = "price", "命中价格规则"
        elif _line_matches_fact(text, promotions):
            role, reason = "promotion", "命中促销规则"
        elif _line_matches_fact(text, ctas):
            role, reason = "cta", "命中行动号召规则"
        elif _line_matches_fact(text, event_dt.get("dates", [])):
            role, reason = "event_time", "命中日期规则"
        elif _line_matches_fact(text, event_dt.get("times", [])):
            role, reason = "event_time", "命中时间规则"
        elif _line_matches_fact(text, locations):
            role, reason = "event_location", "命中地点规则"
        else:
            score = _brand_score(text, index, geom, lines_no_watermark, possible_brand_words)
            assessment = _brand_language_assessment(
                text,
                scene_type=scene_type,
                lines=lines_no_watermark,
                possible_brand_words=possible_brand_words,
                score=score,
                geometry=geom,
            )
            status_rank = {"confirmed": 0, "candidate": 1, "rejected": 2}[assessment["status"]]
            if assessment["status"] != "rejected" and score >= 4.8:
                brand_scored.append((status_rank, score, text, index, assessment))
            # Non-factual text role is assigned after brand winner is selected.
            if float(geom.get("area_ratio", 0.0)) >= 0.012 or float(geom.get("height_ratio", 0.0)) >= 0.035:
                role, reason = "title", "版面中较醒目的文字"
            elif len(text) <= 60 and (text.isupper() or re.search(r"[.!]$", text)):
                role, reason = "slogan", "短句或强调文案"
            else:
                role, reason = "body_text", "普通说明文字"
        records.append({
            "text": text,
            "role": role,
            "confidence": item.get("confidence"),
            "geometry": geom,
            "reason": reason,
            "index": index,
        })

    brand_scored.sort(key=lambda item: (item[0], -item[1], item[3], len(item[2])))
    brand_candidates: List[Dict[str, Any]] = []
    seen = set()
    for status_rank, score, text, index, assessment in brand_scored:
        display = _brand_display(text, lines_no_watermark)
        key = normalize_text(display)
        if not key or key in seen:
            continue
        seen.add(key)
        brand_candidates.append({
            "text": display,
            "raw_text": text,
            "score": round(score, 3),
            "status": assessment.get("status", "candidate"),
            "positive_evidence": assessment.get("positive_evidence", []),
            "negative_evidence": assessment.get("negative_evidence", []),
            "source": "ocr_geometry_linguistic_role_rules",
            "index": index,
        })
        if len(brand_candidates) >= 8:
            break

    confirmed = [item for item in brand_candidates if item.get("status") == "confirmed"]
    winner = confirmed[0] if confirmed else None
    top_candidate = brand_candidates[0] if brand_candidates else None
    if winner:
        winner_norm = normalize_text(winner.get("raw_text"))
        for record in records:
            if normalize_text(record.get("text")) == winner_norm:
                record["role"] = "brand"
                record["reason"] = "品牌式文字通过版面与语言双重校验"
                break

    return {
        "roles": records,
        "brand_candidates": brand_candidates,
        "brand_text": winner.get("text", "") if winner else "",
        "brand_status": "confirmed" if winner else ("candidate" if top_candidate else "none"),
        "brand_candidate_text": top_candidate.get("text", "") if top_candidate and not winner else "",
        "brand_confidence": round(min(0.99, 0.45 + 0.06 * float(winner.get("score", 0.0))), 3) if winner else 0.0,
        "cta_text": ctas,
        "cta_raw_text": cta_raw_text,
        "cta_canonical": cta_canonical,
        "cta_evidence": cta_evidence,
        "direct_cta_text": direct_ctas,
        "direct_cta_raw_text": direct_raw_text,
        "direct_cta_canonical": direct_canonical,
        "direct_cta_evidence": direct_evidence,
        "contextual_cta_text": contextual_ctas,
        "contextual_cta_raw_text": contextual_raw_text,
        "contextual_cta_canonical": contextual_canonical,
        "contextual_cta_evidence": contextual_evidence,
        "price_text": prices,
        "promotion_words": promotions,
        "event_dates": event_dt.get("dates", []),
        "event_times": event_dt.get("times", []),
        "event_locations": locations,
        "canvas": {"width": round(width, 2), "height": round(height, 2)},
    }
