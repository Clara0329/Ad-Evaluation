from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from modules.ocr_module import analyze_ocr
from modules.visual_module import analyze_visual_features
from modules.qwen_module import analyze_image_structured
from modules.evidence_validator import validate_evidence
from modules.score_engine import calculate_scores


PIPELINE_VERSION = "mvp_ocr_qwen_validator_score_v1.3.7.1"
EXPERIMENTAL_MODULE_VERSION = "transalnet_v1.4.3-attention-parallel"



def calculate_center_focus_candidate(
    center_saliency_ratio,
    lower: float = 0.2723,
    upper: float = 0.7656,
):
    """
    将TranSalNet中心显著度转换为0—1候选值。

    lower和upper来自当前288张图片的P5和P95。
    本候选值只用于A/B验证，暂不进入正式评分。
    """
    try:
        value = float(center_saliency_ratio)
    except (TypeError, ValueError):
        return None

    if upper <= lower:
        return None

    normalized = (value - lower) / (upper - lower)
    normalized = max(0.0, min(1.0, normalized))

    return round(normalized, 4)


def _safe_name(value: str) -> str:
    text = re.sub(r"[^0-9A-Za-z._-]+", "_", str(value or "").strip())
    return text.strip("._-") or "unknown"


def _result_stem(image_path: str) -> str:
    path = Path(image_path)
    parts = list(path.parts)
    try:
        index = parts.index("AIGC_images")
    except ValueError:
        return _safe_name(path.stem)
    if len(parts) >= index + 4:
        model = _safe_name(parts[index + 1])
        category = _safe_name(parts[index + 2])
        return f"{model}_{category}_{_safe_name(path.stem)}"
    return _safe_name(path.stem)


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_one(
    image_path: str,
    output_dir: str = "outputs/json_results",
    use_qwen: bool = False,
    use_sam: bool = False,
    use_transalnet: bool = False,
    transalnet_model: str = "models/transalnet/TranSalNet_Res.pth",
    transalnet_device: str = "auto",
) -> Tuple[Dict[str, Any], str]:
    image_file = Path(image_path)
    if not image_file.exists():
        raise FileNotFoundError(f"找不到图片：{image_path}")
    os.makedirs(output_dir, exist_ok=True)

    timings: Dict[str, float] = {}
    start_total = time.perf_counter()

    start = time.perf_counter()
    ocr_result = analyze_ocr(image_path)
    timings["ocr_seconds"] = round(time.perf_counter() - start, 4)

    start = time.perf_counter()
    visual_result = analyze_visual_features(image_path)
    timings["visual_seconds"] = round(time.perf_counter() - start, 4)

    sam_result: Dict[str, Any] = {}
    if use_sam:
        start = time.perf_counter()
        try:
            from modules.sam_module import analyze_visual_focus
            sam_result = analyze_visual_focus(image_path)
        except Exception as exc:
            sam_result = {"error": f"SAM failed: {type(exc).__name__}: {exc}"}
        timings["sam_seconds"] = round(time.perf_counter() - start, 4)

    saliency_result: Dict[str, Any] = {}
    if use_transalnet:
        start = time.perf_counter()
        try:
            from modules.transalnet_module import analyze_saliency
            saliency_result = analyze_saliency(
                image_path=image_path,
                ocr_result=ocr_result,
                output_dir=str(Path(output_dir) / "saliency"),
                output_stem=_result_stem(image_path),
                model_path=transalnet_model,
                device=transalnet_device,
                save_visualizations=True,
                strict=False,
            )
        except Exception as exc:
            saliency_result = {
                "status": "failed",
                "experimental": True,
                "scoring_integration": False,
                "error": f"TranSalNet failed: {type(exc).__name__}: {exc}",
            }

        # TranSalNet中心显著度候选替换值
        if (
            isinstance(saliency_result, dict)
            and saliency_result.get("status") == "ok"
        ):
            center_ratio = (
                saliency_result
                .get("metrics", {})
                .get("center_saliency_ratio")
            )
        
            legacy_center = (
                visual_result.get("saliency_center")
                if isinstance(visual_result, dict)
                else None
            )
        
            center_candidate = (
                calculate_center_focus_candidate(center_ratio)
            )
        
            saliency_result["candidate_features"] = {
                "legacy_saliency_center": legacy_center,
                "center_saliency_ratio": center_ratio,
                "center_focus_candidate": center_candidate,
                "normalization": {
                    "method": "robust_minmax",
                    "lower_percentile": "P5",
                    "upper_percentile": "P95",
                    "lower": 0.2723,
                    "upper": 0.7656,
                    "sample_size": 288,
                },
                "scoring_integration": False,
                "note": (
                    "候选替换值，仅用于与旧saliency_center"
                    "进行A/B验证，尚未进入五维评分。"
                ),
            }

        timings["transalnet_seconds"] = round(time.perf_counter() - start, 4)

    start = time.perf_counter()
    qwen_result = analyze_image_structured(
        image_path=image_path,
        ocr_result=ocr_result,
        visual_result=visual_result,
        use_qwen=use_qwen,
    )
    timings["qwen_seconds"] = round(time.perf_counter() - start, 4)

    start = time.perf_counter()
    evidence = validate_evidence(
        ocr_result=ocr_result,
        visual_result=visual_result,
        qwen_result=qwen_result,
        sam_result=sam_result,
        image_path=image_path,
    )
    timings["validator_seconds"] = round(time.perf_counter() - start, 4)

    start = time.perf_counter()
    score_result = calculate_scores(evidence)

    # v1.4.3 TranSalNet注意力并行候选结果
    attention_candidate_result = {
        "status": "disabled",
        "module_version": (
            "transalnet_v1.4.3-attention-parallel"
        ),
        "scoring_integration": False,
    }

    if (
        use_transalnet
        and isinstance(saliency_result, dict)
        and saliency_result.get("status") == "ok"
    ):
        try:
            from modules.attention_candidate_module import (
                build_attention_candidate,
            )
    
            attention_candidate_result = (
                build_attention_candidate(
                    validated_evidence=evidence,
                    official_score_result=score_result,
                    saliency_analysis=saliency_result,
                )
            )
        except Exception as exc:
            attention_candidate_result = {
                "status": "error",
                "module_version": (
                    "transalnet_v1.4.3-attention-parallel"
                ),
                "scoring_integration": False,
                "error": str(exc),
            }

    timings["score_seconds"] = round(time.perf_counter() - start, 4)
    timings["total_seconds"] = round(time.perf_counter() - start_total, 4)

    result_id = _result_stem(image_path)
    final_result = {
        "result_id": result_id,
        "image_path": image_path,
        "image_sha256": _sha256(image_path),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "pipeline_version": PIPELINE_VERSION,
        "runtime": timings,
        "ocr": ocr_result,
        "visual_features": visual_result,
        "saliency_analysis": saliency_result,
        "experimental_modules": {
            "version": EXPERIMENTAL_MODULE_VERSION,
            "transalnet_enabled": bool(use_transalnet),
            "affects_understanding": False,
            "affects_scoring": False,
        },
        "qwen_analysis": qwen_result,
        "validated_evidence": evidence,
        "attention_candidate": attention_candidate_result,
        "score_result": score_result,
    }

    out_path = os.path.join(output_dir, f"{result_id}_result.json")
    with open(out_path, "w", encoding="utf-8") as file:
        json.dump(final_result, file, ensure_ascii=False, indent=2)
    return final_result, out_path


def _read_image_list(path: str) -> List[str]:
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(f"找不到图片列表：{path}")
    images: List[str] = []
    for raw in file.read_text(encoding="utf-8-sig").splitlines():
        value = raw.strip()
        if not value or value.startswith("#"):
            continue
        images.append(value)
    return images


def run_batch(
    image_paths: Iterable[str],
    output_dir: str,
    use_qwen: bool,
    use_sam: bool,
    skip_existing: bool,
    fail_fast: bool,
    use_transalnet: bool = False,
    transalnet_model: str = "models/transalnet/TranSalNet_Res.pth",
    transalnet_device: str = "auto",
) -> Dict[str, Any]:
    images = list(image_paths)
    started = time.perf_counter()
    records: List[Dict[str, Any]] = []

    for index, image_path in enumerate(images, start=1):
        result_id = _result_stem(image_path)
        expected = Path(output_dir) / f"{result_id}_result.json"
        print(f"\n[{index}/{len(images)}] {image_path}")
        if skip_existing and expected.exists():
            print(f"跳过已有结果：{expected}")
            records.append({"image_path": image_path, "status": "skipped", "output": str(expected)})
            continue
        try:
            result, output = run_one(
                image_path=image_path,
                output_dir=output_dir,
                use_qwen=use_qwen,
                use_sam=use_sam,
                use_transalnet=use_transalnet,
                transalnet_model=transalnet_model,
                transalnet_device=transalnet_device,
            )
            total = result.get("runtime", {}).get("total_seconds")
            print(f"完成：{output}，耗时 {total}s")
            records.append({
                "image_path": image_path,
                "status": "ok",
                "output": output,
                "seconds": total,
            })
        except Exception as exc:
            print(f"失败：{type(exc).__name__}: {exc}")
            records.append({
                "image_path": image_path,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            })
            if fail_fast:
                raise

    summary = {
        "pipeline_version": PIPELINE_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "total_images": len(images),
        "ok": sum(item["status"] == "ok" for item in records),
        "skipped": sum(item["status"] == "skipped" for item in records),
        "failed": sum(item["status"] == "failed" for item in records),
        "batch_seconds": round(time.perf_counter() - started, 4),
        "records": records,
    }
    os.makedirs(output_dir, exist_ok=True)
    summary_path = Path(output_dir) / f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n批处理汇总：{summary_path}")
    print(json.dumps({k: summary[k] for k in ("total_images", "ok", "skipped", "failed", "batch_seconds")}, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--image", help="单张输入图片")
    group.add_argument("--list", dest="image_list", help="每行一个图片路径的批处理列表")
    parser.add_argument("--output", default="outputs/json_results", help="输出目录")
    parser.add_argument("--use-qwen", action="store_true", help="调用Qwen模型")
    parser.add_argument("--use-sam", action="store_true", help="启用SAM")
    parser.add_argument("--use-transalnet", action="store_true", help="启用实验性TranSalNet独立显著性模块；不改变理解层和评分")
    parser.add_argument("--transalnet-model", default="models/transalnet/TranSalNet_Res.pth", help="TranSalNet_Res权重路径")
    parser.add_argument("--transalnet-device", default="auto", help="auto、cpu或cuda:0")
    parser.add_argument("--skip-existing", action="store_true", help="批处理时跳过已有JSON")
    parser.add_argument("--fail-fast", action="store_true", help="批处理遇错立即停止")
    args = parser.parse_args()

    if args.image_list:
        run_batch(
            _read_image_list(args.image_list),
            output_dir=args.output,
            use_qwen=args.use_qwen,
            use_sam=args.use_sam,
            use_transalnet=args.use_transalnet,
            transalnet_model=args.transalnet_model,
            transalnet_device=args.transalnet_device,
            skip_existing=args.skip_existing,
            fail_fast=args.fail_fast,
        )
        return

    image = args.image or "images/test1.jpg"
    result, out_path = run_one(
        image,
        args.output,
        use_qwen=args.use_qwen,
        use_sam=args.use_sam,
        use_transalnet=args.use_transalnet,
        transalnet_model=args.transalnet_model,
        transalnet_device=args.transalnet_device,
    )
    print("\n========== MVP 分析完成 ==========")
    print("输出文件：", out_path)
    print("运行耗时：", result.get("runtime", {}).get("total_seconds"), "秒")
    saliency = result.get("saliency_analysis", {})
    if args.use_transalnet:
        print("TranSalNet状态：", saliency.get("status"), saliency.get("error") or "")
        if saliency.get("status") == "ok":
            print("显著性叠加图：", saliency.get("overlay_path"))
    print("五维评分：")
    print(json.dumps(result["score_result"]["scores"], ensure_ascii=False, indent=2))
    notes = result.get("validated_evidence", {}).get("validator_notes", [])
    if notes:
        print("\n校验修正：")
        for note in notes:
            print("-", note)


if __name__ == "__main__":
    main()
