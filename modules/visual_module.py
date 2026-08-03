"""Deterministic visual feature extractor V1.3.

The features are descriptive evidence, not semantic ground truth. Existing V1.2
keys are preserved for score-engine compatibility; additional keys support later
analysis and ablation studies.
"""

from __future__ import annotations

from typing import Any, Dict, List

import cv2
import numpy as np


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _dominant_color_labels(hsv: np.ndarray) -> List[str]:
    small = cv2.resize(hsv, (64, 64), interpolation=cv2.INTER_AREA)
    hue = small[:, :, 0].astype(np.float32) * 2.0
    sat = small[:, :, 1].astype(np.float32) / 255.0
    val = small[:, :, 2].astype(np.float32) / 255.0
    mask = (sat >= 0.24) & (val >= 0.12)
    if not np.any(mask):
        return ["neutral"]
    hue = hue[mask]
    weight = sat[mask] * np.maximum(val[mask], 0.25)
    bins = [0, 15, 45, 75, 165, 200, 255, 315, 345, 360]
    labels = ["red", "orange", "yellow", "green", "cyan", "blue", "purple", "pink", "red"]
    scores: Dict[str, float] = {}
    for index, label in enumerate(labels):
        selected = (hue >= bins[index]) & (hue < bins[index + 1])
        scores[label] = scores.get(label, 0.0) + float(weight[selected].sum())
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    total = sum(value for _, value in ordered) or 1.0
    return [label for label, value in ordered if value / total >= 0.10][:3] or [ordered[0][0]]


def analyze_visual_features(image_path: str) -> Dict[str, Any]:
    image = cv2.imread(image_path)
    if image is None:
        return {"error": f"Cannot read image: {image_path}"}

    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    brightness = float(np.mean(gray) / 255.0)
    contrast = float(np.std(gray) / 128.0)
    saturation = float(np.mean(hsv[:, :, 1]) / 255.0)

    edges = cv2.Canny(gray, 80, 160)
    edge_density = float(np.mean(edges > 0))

    x1, x2 = int(w * 0.25), int(w * 0.75)
    y1, y2 = int(h * 0.25), int(h * 0.75)
    center_gray = gray[y1:y2, x1:x2]
    center_edges = edges[y1:y2, x1:x2]
    center_energy = float(np.std(center_gray) / 128.0 + np.mean(center_edges > 0))
    global_energy = float(np.std(gray) / 128.0 + np.mean(edges > 0) + 1e-6)
    center_focus = _clip01(center_energy / global_energy)

    # Focus/sharpness proxy. Normalized to a broad 0-1 range, not a quality label.
    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharpness = _clip01(laplacian_variance / 1500.0)

    # Hasler-Süsstrunk-inspired colorfulness proxy.
    b, g, r = cv2.split(image.astype(np.float32))
    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)
    colorfulness_raw = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))
    colorfulness = _clip01(colorfulness_raw / 120.0)

    # Spectral residual saliency if available, otherwise gradient energy.
    saliency_center = center_focus
    try:
        if hasattr(cv2, "saliency"):
            detector = cv2.saliency.StaticSaliencySpectralResidual_create()
            ok, saliency = detector.computeSaliency(image)
            if ok and saliency is not None:
                center_saliency = float(np.mean(saliency[y1:y2, x1:x2]))
                global_saliency = float(np.mean(saliency) + 1e-6)
                saliency_center = _clip01(center_saliency / global_saliency)
    except Exception:
        pass

    # Border complexity helps distinguish dense poster edges from a centered product.
    border = np.ones_like(edges, dtype=bool)
    border[int(h * 0.15):int(h * 0.85), int(w * 0.15):int(w * 0.85)] = False
    border_complexity = float(np.mean(edges[border] > 0)) if np.any(border) else edge_density

    too_dark = brightness < 0.22
    too_bright = brightness > 0.88
    too_cluttered = edge_density > 0.18
    low_sharpness = sharpness < 0.08

    return {
        "image_size": {"width": int(w), "height": int(h)},
        "aspect_ratio": round(float(w / max(h, 1)), 4),
        "brightness": round(_clip01(brightness), 4),
        "contrast": round(_clip01(contrast), 4),
        "saturation": round(_clip01(saturation), 4),
        "colorfulness": round(colorfulness, 4),
        "sharpness": round(sharpness, 4),
        "edge_density": round(_clip01(edge_density), 4),
        "border_complexity": round(_clip01(border_complexity / 0.18), 4),
        "layout_complexity": round(_clip01(edge_density / 0.18), 4),
        "center_focus": round(center_focus, 4),
        "saliency_center": round(saliency_center, 4),
        "dominant_colors": _dominant_color_labels(hsv),
        "quality_flags": {
            "too_dark": too_dark,
            "too_bright": too_bright,
            "too_cluttered": too_cluttered,
            "low_sharpness": low_sharpness,
        },
        "error": None,
    }
