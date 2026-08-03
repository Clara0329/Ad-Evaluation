from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.transalnet_module import _get_model, _resolve_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/transalnet/TranSalNet_Res.pth")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    model_file = Path(args.model)
    if not model_file.exists():
        raise SystemExit(f"权重不存在：{model_file}")
    if model_file.stat().st_size < 10 * 1024 * 1024:
        raise SystemExit(f"权重文件异常小：{model_file.stat().st_size} bytes")
    device = _resolve_device(args.device)
    _get_model(str(model_file), device)
    print(json.dumps({
        "status": "ok",
        "model": str(model_file),
        "size_mb": round(model_file.stat().st_size / 1024 / 1024, 2),
        "device": device,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
