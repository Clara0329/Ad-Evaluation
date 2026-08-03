"""
Bilingual OCR text semantics for AIGC advertisements (V1.3.7).

This module is deterministic and model-independent. It converts raw OCR lines into
conservative, source-traceable facts used by the validator:
- price / promotion / CTA
- event time / date / location markers
- brand candidates
- text quality status
- scene keyword scores

Design principle: precision before recall. Missing a weak fact is preferable to
accepting a hallucinated brand, price, CTA, or event field.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


_SPACE_RE = re.compile(r"\s+")
_NORMALIZE_RE = re.compile(r"[^0-9a-z\u4e00-\u9fff]+", re.I)


def clean_text(value: Any) -> str:
    return _SPACE_RE.sub(" ", str(value or "")).strip()


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", clean_text(value).lower())
    text = "".join(char for char in text if not unicodedata.combining(char))
    return _NORMALIZE_RE.sub("", text)


def unique_strings(values: Iterable[Any]) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values:
        text = clean_text(value)
        key = normalize_text(text)
        if text and key and key not in seen:
            seen.add(key)
            result.append(text)
    return result


WATERMARK_PATTERNS = (
    re.compile(r"^ai\s*生成$", re.I),
    re.compile(r"^generated\s+by\s+ai$", re.I),
    re.compile(r"^ai\s*generated$", re.I),
)


def is_watermark(text: str) -> bool:
    value = clean_text(text)
    return any(pattern.search(value) for pattern in WATERMARK_PATTERNS)


# Price must contain a currency sign/code or an explicit price label.
PRICE_PATTERNS = (
    re.compile(r"(?:[$€£￥¥])\s*\d+(?:[.,]\d{1,2})?", re.I),
    re.compile(r"\d+(?:[.,]\d{1,2})?\s*(?:usd|cny|rmb|eur|gbp|元|块钱)\b", re.I),
    re.compile(
        r"(?:price|now|only|from|starting\s+at|售价|价格|现价|原价|到手价|优惠价|活动价|促销价)"
        r"\s*[:：]?\s*(?:[$€£￥¥])?\s*\d+(?:[.,]\d{1,2})?\s*(?:usd|cny|rmb|eur|gbp|元|块钱)?",
        re.I,
    ),
)

# Things that look numeric but are not prices.
NON_PRICE_UNITS = re.compile(
    r"(?:mah|mp|gb|tb|hz|khz|mhz|ghz|km|cm|mm|ml|l|pa|w|kw|v|a|days?|hours?|mins?|minutes?|%|inch|\")\b",
    re.I,
)

PROMOTION_PATTERNS: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    ("限时优惠", re.compile(r"限时(?:优惠|特价|抢购)|limited\s*time(?:\s+offer)?", re.I)),
    ("折扣", re.compile(r"\b\d{1,2}\s*%\s*off\b|\b\d(?:\.\d)?\s*折\b|折扣|discount", re.I)),
    ("首单优惠", re.compile(r"first\s+order|首单|首次下单", re.I)),
    ("免费试用", re.compile(r"free\s*trial|免费试用", re.I)),
    ("免费月份", re.compile(r"first\s+month\s+free|首月免费", re.I)),
    ("免加入费", re.compile(r"(?:no|0|o)\s*joining\s*fee|免加入费", re.I)),
    ("会员优惠", re.compile(r"new\s+member\s+deal|member\s+deal|会员优惠", re.I)),
    ("优惠码", re.compile(r"(?:use\s+code\s*[:：]?\s*[a-z0-9_-]+|code\s*[:：]\s*[a-z0-9_-]+)|优惠码|兑换码", re.I)),
    ("买一送一", re.compile(r"buy\s+one\s+get\s+one|bogo|买一送一", re.I)),
    ("满减", re.compile(r"满\s*\d+\s*减\s*\d+|满减", re.I)),
    ("赠品", re.compile(r"free\s+(?:gift|onboarding|session|shake)|赠品|赠送", re.I)),
    ("特价", re.compile(r"\bsale\b|special\s+offer|特价|促销价|秒杀|清仓", re.I)),
)

CTA_PATTERNS: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    ("SHOP NOW", re.compile(r"\bshop\s*now\b", re.I)),
    ("BUY NOW", re.compile(r"\bbuy\s*now\b", re.I)),
    ("GRAB YOURS NOW", re.compile(r"\bgrab\s*yours\s*now\b", re.I)),
    ("JOIN NOW", re.compile(r"\bjoin\s*now(?=\b|\s*start)", re.I)),
    ("JOIN US", re.compile(r"\b(?:come\s*)?join\s*us\b", re.I)),
    ("JOIN THE CHALLENGE", re.compile(r"\bjoin\s*the\s*challenge\b", re.I)),
    ("REGISTER NOW", re.compile(r"\bregister\s*(?:online\s*)?now\b", re.I)),
    ("LEARN MORE", re.compile(r"\blearn\s*morex?\b", re.I)),
    ("ENROLL NOW", re.compile(r"\benroll\s*now\b", re.I)),
    ("ENROLL TODAY", re.compile(r"\benroll\s*today\b", re.I)),
    ("GET STARTED", re.compile(r"\bget\s*started(?:\s*now)?\b", re.I)),
    ("START LEARNING", re.compile(r"\bstart\s*learning(?:\s*today)?\b", re.I)),
    ("START YOUR SUCCESS STORY", re.compile(r"\bstart\s*your\s*success\s*story\b", re.I)),
    ("MUST WATCH", re.compile(r"\bmust\s*watch\b", re.I)),
    ("CLICK NOW", re.compile(r"\bclick\s*now\b", re.I)),
    ("BOOK NOW", re.compile(r"\bbook\s+(?:now|today|your\b[^.!]{0,30})", re.I)),
    ("SIGN UP", re.compile(r"\bsign\s*up\b", re.I)),
    ("SCAN QR", re.compile(r"\bscan\s+(?:the\s+)?qr\b", re.I)),
    ("EXPERIENCE THE FUTURE", re.compile(r"\bexperience\s*the\s*future\b", re.I)),
    ("PROTECT NATURE", re.compile(r"\bprotect\s*nature\b", re.I)),
    ("REDUCE & REUSE", re.compile(r"\breduce\s*(?:&|and)?\s*reuse\b", re.I)),
    ("BUILD A BETTER TOMORROW", re.compile(r"\b(?:let'?s\s*)?build\s*a\s*better\s*tomorrow(?:\s*,?\s*together)?\b", re.I)),
    ("立即购买", re.compile(r"立即购买|马上购买|点击购买|立即抢购|马上抢|下单", re.I)),
    ("立即报名", re.compile(r"立即报名|马上报名|扫码报名|报名参加", re.I)),
    ("了解更多", re.compile(r"了解更多|点击查看|点击观看", re.I)),
    ("预约", re.compile(r"立即预约|预约咨询|预约直播", re.I)),
)

COMPACT_CTA_PATTERNS: Tuple[Tuple[str, str], ...] = (
    ("GRAB YOURS NOW", "grabyoursnow"),
    ("JOIN NOW", "joinnow"),
    ("JOIN US", "comejoinus"),
    ("JOIN THE CHALLENGE", "jointhechallenge"),
    ("REGISTER NOW", "registernow"),
    ("LEARN MORE", "learnmore"),
    ("ENROLL NOW", "enrollnow"),
    ("ENROLL TODAY", "enrolltoday"),
    ("START LEARNING", "startlearningtoday"),
    ("START YOUR SUCCESS STORY", "startyoursuccessstory"),
    ("MUST WATCH", "mustwatch"),
    ("CLICK NOW", "clicknow"),
    ("SHOP NOW", "shopnow"),
    ("BUY NOW", "buynow"),
    ("GET STARTED", "getstarted"),
    ("EXPERIENCE THE FUTURE", "experiencethefuture"),
    ("PROTECT NATURE", "protectnature"),
    ("REDUCE & REUSE", "reducereuse"),
    ("BUILD A BETTER TOMORROW", "buildabettertomorrow"),
)


# These phrases are likely headings, claims, or UI roles rather than brands.
GENERIC_BRAND_TERMS = {
    "premium", "prestige", "luxury", "natural radiance", "shop now",
    "learn more", "join now", "join us", "get started", "enroll now",
    "trusted by", "master in", "hours", "tropical", "paradise",
    "your dream", "drive the", "join the", "our earth", "our planet",
    "new member deal", "limited time offer", "coming soon", "club",
    "university", "fitness", "coffee", "skincare", "performance",
    "education", "technology", "innovation", "future", "sale",
    "the future", "the future of", "learn new skills", "make short videos",
    "ai生成", "generated by ai", "ai generated",
}

BRAND_ROLE_REJECT_PATTERNS = (
    re.compile(r"^(?:shop|buy|join|learn|enroll|get|start|book|scan|click|discover|explore)\b", re.I),
    re.compile(r"\b(?:off|free\s*trial|deal|offer|sale|discount)\b", re.I),
    re.compile(r"^(?:when|where|date|time|location|event details|trusted by)\b", re.I),
    re.compile(r"(?:your|our|the)\s+(?:future|dream|story|planet|earth|pace|speed|goals)$", re.I),
    re.compile(r"[-–—]$"),
)

# Scene signal dictionaries. Specific categories are evaluated before generic brand/ecommerce.
SCENE_PATTERNS: Mapping[str, Tuple[re.Pattern[str], ...]] = {
    "教育校园宣传": (
        re.compile(r"\b(?:course|courses|learn|learning|skills?|curriculum|certificate|enroll|student|students|university|campus|club fair|career|job-ready|data science|web development|exam prep|pass rate)\b", re.I),
        re.compile(r"教育|培训|课程|学习|招生|校园|大学|社团|公开课|讲座|证书|职业技能|考试", re.I),
    ),
    "公益宣传": (
        re.compile(r"\b(?:our planet|our earth|responsibility|sustainable future|protect nature|reduce\s*&?\s*reuse|lasting change|better tomorrow|clean energy|protect the planet|health awareness|prevention|well-being|stay healthy|healthy today|healthier community|get vaccinated|wash hands|care for others)\b", re.I),
        re.compile(r"公益|环保|保护环境|关爱|倡议|志愿者|捐赠|可持续|地球|责任|健康宣传|预防|社区健康", re.I),
    ),
    "短视频封面": (
        re.compile(r"\b(?:short videos?|video|tutorial|tips\s*&\s*tricks|for beginners|secret trick|\d+\s*steps?|in\s*\d+\s*days?|no fluff|day\s*\d+|must\s*watch|trend alert|trending|viral|click\s*now)\b", re.I),
        re.compile(r"短视频|视频封面|教程|干货|探店|种草|测评视频|知识分享|新手|必看|热门|趋势", re.I),
    ),
    "旅游宣传": (
        re.compile(r"\b(?:tropical|island|paradise|ocean|beach|resort|hotel|vacation|holiday|getaway|escape|serenity|crystal waters|sunset|travel|destination|landmarks?|heritage|city charm|explore\s*[.&]\s*discover)\b", re.I),
        re.compile(r"旅游|旅行|度假|海岛|海滩|沙滩|酒店|度假村|景区|景点|目的地|海景|泳池|城市风光|文化遗产", re.I),
    ),
    "活动宣传海报": (
        re.compile(r"\b(?:event details|date|time|location|when|where|festival|conference|summit|fair|open day|student union|main quad|hackathon|challenge|competition|prizes?|register\s*now|date\s*&\s*venue|teams?\s*of|top team|room\s*\d+)\b", re.I),
        re.compile(r"活动时间|活动地点|直播时间|发布会|购物节|音乐节|展览|赛事|比赛|竞赛|挑战赛|晚会|招新|峰会|报名|奖项", re.I),
    ),
    "科技创新宣传": (
        re.compile(r"\b(?:research|innovation|scientific|breakthrough|patented|peer-reviewed|project name|next-gen|advanced materials|publications?|citations?|research team|technology|pioneering solutions|rigorous research|future impact)\b", re.I),
        re.compile(r"科研|研究项目|科技创新|技术创新|科学研究|专利|论文|突破|实验室|未来技术", re.I),
    ),
}


EVENT_DATE_PATTERNS = (
    re.compile(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s*\d{1,2}(?:\s*[-–]\s*\d{1,2})?(?:,?\s*20\d{2})?\b", re.I),
    re.compile(r"\b20\d{2}[/-]\d{1,2}[/-]\d{1,2}\b"),
    re.compile(r"\d{1,2}月\d{1,2}日"),
)
EVENT_TIME_PATTERNS = (
    re.compile(r"\b\d{1,2}:\d{2}\s*(?:am|pm)?(?:\s*[-–]\s*\d{1,2}:\d{2}\s*(?:am|pm)?)?\b", re.I),
    re.compile(r"\b\d{1,2}\s*(?:am|pm)\s*[-–]\s*\d{1,2}\s*(?:am|pm)\b", re.I),
)
LOCATION_MARKERS = (
    "location", "where", "venue", "student union", "main quad", "auditorium",
    "地点", "地址", "会场", "礼堂", "体育馆",
)

# Common English vocabulary used only as a weak signal for obviously corrupted text.
# It is intentionally small and conservative; the VLM remains the primary source for typo judgement.
COMMON_WORDS = {
    "a", "ai", "all", "and", "any", "are", "at", "be", "beauty", "better", "bold",
    "book", "brand", "build", "by", "care", "career", "clean", "club", "coffee", "come",
    "connect", "create", "data", "day", "design", "dream", "earth", "education", "efficiency",
    "energy", "enjoy", "escape", "event", "experience", "expert", "fitness", "for", "free",
    "fresh", "from", "future", "get", "go", "grow", "hand", "in", "innovation", "intelligence",
    "island", "join", "learn", "life", "limited", "make", "master", "member", "more", "natural",
    "new", "no", "now", "of", "off", "offer", "online", "our", "pace", "paradise", "planet",
    "premium", "protect", "ready", "responsibility", "results", "roasted", "shop", "skills", "smart",
    "start", "strong", "style", "support", "sustainable", "the", "today", "train", "training",
    "trusted", "unleash", "unlimited", "us", "video", "videos", "with", "work", "your",
}


def _extract_pattern_matches(text: str, patterns: Sequence[re.Pattern[str]]) -> List[str]:
    values: List[str] = []
    for pattern in patterns:
        for match in pattern.finditer(text):
            value = clean_text(match.group(0))
            if value and value not in values:
                values.append(value)
    return values


def extract_prices(lines: Sequence[str]) -> List[str]:
    """Extract commercial prices while excluding prizes, awards and specs."""
    result: List[str] = []
    reward_markers = re.compile(
        r"\b(?:prize|prizes|award|awards|reward|rewards|winner|winning|top team|grand prize|cash prize)\b|奖金|奖励|奖品|冠军",
        re.I,
    )
    explicit_price_label = re.compile(
        r"\b(?:price|now|only|from|starting at|sale price)\b|售价|价格|现价|原价|到手价|优惠价|活动价|促销价",
        re.I,
    )
    all_text = " ".join(lines)
    global_reward_context = bool(reward_markers.search(all_text))
    for index, line in enumerate(lines):
        context = " ".join(lines[max(0, index - 5): min(len(lines), index + 6)])
        for price_pattern in PRICE_PATTERNS:
            for match in price_pattern.finditer(line):
                value = clean_text(match.group(0))
                if NON_PRICE_UNITS.search(value) and not re.search(r"[$€£￥¥]|\b(?:usd|cny|rmb|eur|gbp|元)\b", value, re.I):
                    continue
                if (reward_markers.search(context) or global_reward_context) and not explicit_price_label.search(line):
                    continue
                if value and value not in result:
                    result.append(value)
    return result


def _text_windows(lines: Sequence[str]) -> List[str]:
    values = list(lines)
    for index in range(len(lines) - 1):
        values.append(clean_text(lines[index] + " " + lines[index + 1]))
    return unique_strings(values)


def extract_promotions(lines: Sequence[str]) -> List[str]:
    found: List[str] = []
    for candidate in _text_windows(lines):
        for label, pattern in PROMOTION_PATTERNS:
            match = pattern.search(candidate)
            if match:
                value = clean_text(match.group(0)) or label
                if value not in found:
                    found.append(value)
    return found


def extract_ctas(lines: Sequence[str]) -> List[str]:
    """Extract and normalize direct action phrases, including compact OCR text."""
    found: List[str] = []
    candidates = _text_windows(lines)
    for canonical, pattern in CTA_PATTERNS:
        if any(pattern.search(candidate) for candidate in candidates):
            found.append(canonical)

    compact_candidates = [normalize_text(candidate) for candidate in candidates]
    for canonical, compact_pattern in COMPACT_CTA_PATTERNS:
        if any(compact_pattern in compact for compact in compact_candidates):
            found.append(canonical)

    # JOIN NOW embedded before START YOUR... contains two valid CTAs.
    return unique_strings(found)


def extract_event_date_time(lines: Sequence[str]) -> Dict[str, List[str]]:
    """Extract dates/times and repair OCR-fragmented month/date pairs."""
    cleaned = [clean_text(line) for line in lines if clean_text(line)]
    candidates = unique_strings(_text_windows(cleaned) + [" ".join(cleaned), " | ".join(cleaned)])
    dates: List[str] = []
    times: List[str] = []
    for candidate in candidates:
        dates.extend(_extract_pattern_matches(candidate, EVENT_DATE_PATTERNS))
        times.extend(_extract_pattern_matches(candidate, EVENT_TIME_PATTERNS))

    month_only = re.compile(r"^(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*$", re.I)
    date_tail = re.compile(r"^\s*(\d{1,2}(?:\s*[-–]\s*\d{1,2})?(?:,?\s*20\d{2})?)\s*$", re.I)
    for index, line in enumerate(cleaned):
        if not month_only.fullmatch(line):
            continue
        for nxt in cleaned[index + 1: index + 7]:
            match = date_tail.fullmatch(nxt)
            if match:
                dates.append(clean_text(f"{line} {match.group(1)}"))
                break
    return {"dates": unique_strings(dates), "times": unique_strings(times)}


def extract_locations(lines: Sequence[str]) -> List[str]:
    """Extract event venues while tolerating OCR column fragmentation."""
    result: List[str] = []
    exact_labels = {"location", "venue", "地点", "地址", "会场", "datevenue"}
    location_words = re.compile(
        r"\b(?:hall|room\s*\d+|campus|auditorium|sports\s+complex|student\s+union|main\s+quad)\b|体育馆|礼堂|会场|大厅|教室",
        re.I,
    )
    modifier_words = re.compile(r"^(?:engineering|science|student)$", re.I)
    cleaned = [clean_text(line).strip(" ?:：&") for line in lines if clean_text(line)]

    for index, line in enumerate(cleaned):
        match = re.search(r"(?:location|venue|地点|地址|会场)\s*[:：]\s*(.+)$", line, re.I)
        if match:
            value = clean_text(match.group(1)).strip(" ?:：")
            if len(normalize_text(value)) >= 2:
                result.append(value)
        if normalize_text(line) in exact_labels and index + 1 < len(cleaned):
            value = clean_text(cleaned[index + 1]).strip(" ?:：")
            if len(normalize_text(value)) >= 2 and normalize_text(value) not in exact_labels:
                result.append(value)

    has_specific_campus = any(re.search(r"\bmain\s+campus\b", line, re.I) for line in cleaned)
    has_hall_or_room = any(re.search(r"\b(?:hall|room\s*\d+)\b", line, re.I) for line in cleaned)
    venue_parts: List[str] = []
    for line in cleaned:
        norm = normalize_text(line)
        if norm in exact_labels or norm == "datevenue":
            continue
        if has_specific_campus and re.fullmatch(r"campus", line, re.I):
            continue
        if location_words.search(line):
            venue_parts.append(line)
        elif has_hall_or_room and modifier_words.fullmatch(line):
            venue_parts.append(line)

    venue_parts = unique_strings(venue_parts)
    if venue_parts:
        combined = clean_text(" ".join(venue_parts)).strip(" ?:：&")
        combined = re.sub(r"\bAuditorium\s+Sports Complex\b", "Auditorium & Sports Complex", combined, flags=re.I)
        if len(normalize_text(combined)) >= 3:
            result.insert(0, combined)

    for line in venue_parts:
        if len(normalize_text(line)) >= 2:
            result.append(line)
    return unique_strings(result)[:6]


def _tokenize_english(text: str) -> List[str]:
    return re.findall(r"[A-Za-z]{2,}", text)


def suspicious_text_lines(lines: Sequence[str]) -> List[str]:
    """Conservative detector for likely garbled/unfinished OCR strings.

    It deliberately does not classify every unknown brand as garbled. It focuses on
    visible structural anomalies: dangling punctuation, mixed symbol fragments,
    duplicated terminal letters, and long all-letter strings with multiple unlikely
    segments. Qwen's image-level judgement can add more candidates later.
    """
    suspicious: List[str] = []
    for line in lines:
        text = clean_text(line)
        if not text or is_watermark(text):
            continue
        norm = normalize_text(text)
        if len(norm) < 3:
            continue

        reasons = 0
        if re.search(r"[-–—]$", text):
            reasons += 2
        if re.search(r"[?]{2,}|[<>{}\[\]\\]{2,}", text):
            reasons += 2
        if re.search(r"\b[A-Z]{3,}[0-9]+[A-Z]+\b", text):
            reasons += 1
        tokens = _tokenize_english(text)
        if tokens:
            unknown_long = [
                token.lower() for token in tokens
                if len(token) >= 7 and token.lower() not in COMMON_WORDS
            ]
            # Only use this when several long unknown tokens occur, avoiding brand false positives.
            if len(unknown_long) >= 3:
                reasons += 1
        if reasons >= 2:
            suspicious.append(text)
    return unique_strings(suspicious)


def infer_text_status(lines: Sequence[str], suspicious: Sequence[str]) -> str:
    meaningful = [line for line in lines if normalize_text(line) and not is_watermark(line)]
    if not meaningful:
        return "absent"
    suspicious_keys = {normalize_text(x) for x in suspicious}
    suspicious_count = sum(normalize_text(line) in suspicious_keys for line in meaningful)
    ratio = suspicious_count / max(len(meaningful), 1)
    if suspicious_count == 0:
        return "readable"
    if ratio >= 0.45 or suspicious_count >= 4:
        return "garbled"
    return "partially_readable"


def _is_generic_brand_phrase(text: str) -> bool:
    norm_space = clean_text(text).lower().strip(" .,:;!?>→")
    norm = normalize_text(norm_space)
    if not norm:
        return True
    if norm_space in GENERIC_BRAND_TERMS:
        return True
    if is_watermark(text):
        return True
    if any(pattern.search(norm_space) for pattern in BRAND_ROLE_REJECT_PATTERNS):
        return True
    if re.search(r"\b(?:off|free|trial|deal|offer|price|date|time|location)\b", norm_space, re.I):
        return True
    return False


def brand_candidate_score(text: str, lines: Sequence[str], qwen_candidate: str = "") -> float:
    value = clean_text(text).strip(" |•·:;,.!?>→")
    norm = normalize_text(value)
    if len(norm) < 3 or len(value) > 28:
        return -100.0
    if _is_generic_brand_phrase(value):
        return -100.0
    if re.fullmatch(r"\d+(?:\.\d+)?", value):
        return -100.0
    if len(value.split()) > 4:
        return -8.0

    score = 0.0
    if qwen_candidate and (
        normalize_text(qwen_candidate) == norm
        or norm in normalize_text(qwen_candidate)
        or normalize_text(qwen_candidate) in norm
    ):
        score += 5.0

    line_norms = [normalize_text(line) for line in lines]
    repeats = sum(1 for item in line_norms if item == norm)
    if repeats >= 2:
        score += 4.0
    elif repeats == 1:
        score += 1.0

    if re.fullmatch(r"[A-Z][A-Z0-9&.-]{2,15}", value):
        score += 2.0
    elif re.fullmatch(r"[A-Za-z][A-Za-z0-9&.' -]{2,20}", value):
        score += 1.5
    elif re.fullmatch(r"[\u4e00-\u9fff]{2,8}", value):
        score += 2.0

    # Concatenated two-token marks such as ECLATBLANC or NEXIONX are plausible.
    if value.isupper() and 4 <= len(value) <= 14:
        score += 1.0
    if lines and norm == normalize_text(lines[0]):
        score += 0.5
    return score


def select_brand_candidates(
    lines: Sequence[str],
    qwen_candidate: str = "",
    provided_candidates: Optional[Sequence[str]] = None,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    pool = unique_strings(list(provided_candidates or []) + list(lines))
    scored: List[Tuple[float, str]] = []
    for candidate in pool:
        score = brand_candidate_score(candidate, lines, qwen_candidate=qwen_candidate)
        if score >= 4.0:
            scored.append((score, clean_text(candidate).strip(" |•·:;,.!?>→")))
    scored.sort(key=lambda item: (-item[0], len(item[1])))
    result: List[Dict[str, Any]] = []
    seen = set()
    for score, value in scored:
        key = normalize_text(value)
        if key in seen:
            continue
        seen.add(key)
        result.append({"text": value, "score": round(score, 2), "source": "ocr"})
        if len(result) >= limit:
            break
    return result


def scene_keyword_scores(lines: Sequence[str], facts: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    joined = " | ".join(lines)
    scores = {scene: 0.0 for scene in SCENE_PATTERNS}
    for scene, patterns in SCENE_PATTERNS.items():
        for pattern in patterns:
            matches = pattern.findall(joined)
            if matches:
                scores[scene] += min(4.0, 1.0 + 0.5 * len(matches))

    facts = facts or {}
    if facts.get("event_dates") or facts.get("event_times") or facts.get("event_locations"):
        scores["活动宣传海报"] += 2.5
    if facts.get("cta_text"):
        # CTA alone is not enough for ecommerce; it only provides weak conversion evidence.
        scores.setdefault("电商商品广告", 0.0)
        scores["电商商品广告"] += 0.8
    if facts.get("price_text") or facts.get("promotion_words"):
        scores.setdefault("电商商品广告", 0.0)
        scores["电商商品广告"] += 2.5
    return {key: round(value, 3) for key, value in scores.items()}


def analyze_text_semantics(
    ocr_result: Dict[str, Any],
    qwen_brand_candidate: str = "",
    qwen_suspicious_text: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    raw_lines = [clean_text(x) for x in (ocr_result.get("all_text", []) or [])]
    lines = unique_strings(line for line in raw_lines if line and not is_watermark(line))

    prices = extract_prices(lines)
    promotions = extract_promotions(lines)
    ctas = extract_ctas(lines)
    event_dt = extract_event_date_time(lines)
    locations = extract_locations(lines)

    deterministic_suspicious = suspicious_text_lines(lines)
    qwen_supported_suspicious = [
        text for text in unique_strings(qwen_suspicious_text or [])
        if normalize_text(text) and normalize_text(text) in normalize_text(" ".join(lines))
    ]
    suspicious = unique_strings(deterministic_suspicious + qwen_supported_suspicious)
    status = infer_text_status(lines, suspicious)

    provided_candidates = ocr_result.get("possible_brand_words", []) or []
    brand_lines = [line for line in raw_lines if line and not is_watermark(line)]
    brands = select_brand_candidates(
        brand_lines,
        qwen_candidate=qwen_brand_candidate,
        provided_candidates=provided_candidates,
    )

    facts = {
        "price_text": prices,
        "promotion_words": promotions,
        "cta_text": ctas,
        "event_dates": event_dt["dates"],
        "event_times": event_dt["times"],
        "event_locations": locations,
    }
    scene_scores = scene_keyword_scores(lines, facts)

    readable = [line for line in lines if normalize_text(line) not in {normalize_text(x) for x in suspicious}]
    return {
        "status": status,
        "has_visible_text": bool(lines),
        "line_count": len(lines),
        "readable_lines": readable[:30],
        "suspicious_lines": suspicious[:12],
        "garbled_ratio": round(len(suspicious) / max(len(lines), 1), 4) if lines else 0.0,
        "price_text": prices,
        "promotion_words": promotions,
        "cta_text": ctas,
        "event_dates": event_dt["dates"],
        "event_times": event_dt["times"],
        "event_locations": locations,
        "brand_candidates": brands,
        "scene_keyword_scores": scene_scores,
        "watermark_removed": any(is_watermark(line) for line in raw_lines),
    }
