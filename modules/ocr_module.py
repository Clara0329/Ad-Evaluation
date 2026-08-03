"""
PaddleOCR wrapper for multilingual AIGC advertisement analysis (V3.0).

Key changes from V2:
- bilingual Chinese/English price, promotion and CTA extraction;
- conservative brand candidates;
- OCR confidence preservation when available;
- deterministic text-quality and text-role analysis;
- one cached OCR instance per process.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

from modules.text_semantics import analyze_text_semantics, clean_text


_ocr_instance = None


def _get_ocr():
    global _ocr_instance
    if _ocr_instance is not None:
        return _ocr_instance

    from paddleocr import PaddleOCR

    lang = os.environ.get("AIGC_OCR_LANG", "ch")
    try:
        _ocr_instance = PaddleOCR(
            use_textline_orientation=True,
            lang=lang,
            use_gpu=False,
        )
    except TypeError:
        _ocr_instance = PaddleOCR(
            use_angle_cls=True,
            lang=lang,
            use_gpu=False,
        )
    return _ocr_instance


def _flatten_paddle_result(results: Any) -> Tuple[List[str], List[Dict[str, Any]]]:
    texts: List[str] = []
    items: List[Dict[str, Any]] = []
    if not results:
        return texts, items

    if not isinstance(results, list):
        results = [results]

    for res in results:
        if isinstance(res, dict) and "rec_texts" in res:
            rec_texts = list(res.get("rec_texts", []) or [])
            rec_scores = list(res.get("rec_scores", []) or [])
            rec_boxes = list(res.get("rec_boxes", []) or [])
            for index, raw_text in enumerate(rec_texts):
                text = clean_text(raw_text)
                if not text:
                    continue
                score = None
                if index < len(rec_scores):
                    try:
                        score = round(float(rec_scores[index]), 4)
                    except (TypeError, ValueError):
                        score = None
                box = rec_boxes[index] if index < len(rec_boxes) else None
                texts.append(text)
                items.append({"text": text, "confidence": score, "box": box})
            continue

        if isinstance(res, list):
            for item in res:
                try:
                    text = clean_text(item[1][0])
                    if not text:
                        continue
                    score = round(float(item[1][1]), 4)
                    box = item[0]
                    texts.append(text)
                    items.append({"text": text, "confidence": score, "box": box})
                except Exception:
                    continue

    return texts, items


def _base_error(exc: Exception) -> Dict[str, Any]:
    return {
        "all_text": [],
        "text_items": [],
        "joined_text": "",
        "has_price": False,
        "price_text": [],
        "has_cta": False,
        "cta_text": [],
        "has_promotion": False,
        "promotion_words": [],
        "possible_brand_words": [],
        "text_analysis": {
            "status": "absent",
            "has_visible_text": False,
            "line_count": 0,
            "readable_lines": [],
            "suspicious_lines": [],
            "garbled_ratio": 0.0,
            "price_text": [],
            "promotion_words": [],
            "cta_text": [],
            "event_dates": [],
            "event_times": [],
            "event_locations": [],
            "brand_candidates": [],
            "scene_keyword_scores": {},
            "watermark_removed": False,
        },
        "error": f"OCR failed: {type(exc).__name__}: {exc}",
    }


def detect_text(image_path: str) -> List[str]:
    return analyze_ocr(image_path)["all_text"]


def analyze_ocr(image_path: str) -> Dict[str, Any]:
    try:
        ocr = _get_ocr()
        if hasattr(ocr, "predict"):
            raw = ocr.predict(image_path)
        else:
            raw = ocr.ocr(image_path, cls=True)
        texts, text_items = _flatten_paddle_result(raw)
    except Exception as exc:
        return _base_error(exc)

    joined = " ".join(texts)

    # First pass without Qwen. The validator may run a second pass using a
    # Qwen-proposed brand or suspicious-text candidates.
    provisional = {
        "all_text": texts,
        "joined_text": joined,
        "possible_brand_words": [],
    }
    text_analysis = analyze_text_semantics(provisional)
    possible_brand_words = [
        item["text"] for item in text_analysis.get("brand_candidates", [])
    ]

    # Re-run with deterministic candidate list included for stable output.
    provisional["possible_brand_words"] = possible_brand_words
    text_analysis = analyze_text_semantics(provisional)

    return {
        "all_text": texts,
        "text_items": text_items,
        "joined_text": joined,
        "has_price": bool(text_analysis["price_text"]),
        "price_text": text_analysis["price_text"],
        "has_cta": bool(text_analysis["cta_text"]),
        "cta_text": text_analysis["cta_text"],
        "has_promotion": bool(text_analysis["promotion_words"]),
        "promotion_words": text_analysis["promotion_words"],
        "possible_brand_words": [
            item["text"] for item in text_analysis.get("brand_candidates", [])
        ],
        "text_analysis": text_analysis,
        "error": None,
    }
