from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.ocr_module import analyze_ocr
from modules.transalnet_module import analyze_saliency


def main() -> None:
    parser = argparse.ArgumentParser(description="独立测试TranSalNet，不执行Qwen和五维评分")
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", default="models/transalnet/TranSalNet_Res.pth")
    parser.add_argument("--output", default="outputs/transalnet_smoke")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-ocr", action="store_true")
    args = parser.parse_args()

    ocr_result = {} if args.skip_ocr else analyze_ocr(args.image)
    result = analyze_saliency(
        image_path=args.image,
        ocr_result=ocr_result,
        output_dir=args.output,
        output_stem=Path(args.image).stem,
        model_path=args.model,
        device=args.device,
        strict=False,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get("status") != "ok":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
