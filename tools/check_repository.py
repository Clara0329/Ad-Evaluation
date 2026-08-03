#!/usr/bin/env python3
"""Lightweight repository check that does not load model weights."""

from __future__ import annotations

import compileall
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    ROOT / "main.py",
    ROOT / "modules" / "score_engine.py",
    ROOT / "modules" / "evidence_validator.py",
    ROOT / "modules" / "attention_candidate_module.py",
    ROOT / "configs" / "attention_calibration_v143.json",
]


def main() -> None:
    missing = [str(path.relative_to(ROOT)) for path in REQUIRED if not path.exists()]
    if missing:
        raise SystemExit(f"Missing required files: {missing}")

    if not compileall.compile_dir(str(ROOT / "modules"), quiet=1):
        raise SystemExit("Module compilation failed")
    if not compileall.compile_file(str(ROOT / "main.py"), quiet=1):
        raise SystemExit("main.py compilation failed")

    config = json.loads(
        (ROOT / "configs" / "attention_calibration_v143.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["version"] == "transalnet_v1.4.3-attention-parallel"
    assert config["scoring_integration"] is False
    assert config["training_sample_size"] == 288

    from modules.score_engine import calculate_scores

    evidence = {
        "scene_type": "其他",
        "visual_features": {
            "brightness": 0.5,
            "contrast": 0.5,
            "saturation": 0.5,
            "center_focus": 0.5,
            "layout_complexity": 0.5,
            "edge_density": 0.1,
            "quality_flags": {},
        },
        "selling_points": [],
        "attention_elements": [],
        "risk_points": [],
        "memory_points": [],
        "trust_categories": {},
        "memory_categories": {},
        "emotion_style": {},
    }
    result = calculate_scores(evidence)
    assert set(result["scores"]) == {
        "persuasion",
        "arousal",
        "trust",
        "attention",
        "memory",
        "total",
    }
    print("Repository check passed")
    print("Scoring smoke result:", result["scores"])


if __name__ == "__main__":
    main()
