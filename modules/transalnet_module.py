"""Independent TranSalNet visual-saliency module for the AIGC ad pipeline.

This module is experimental and deliberately isolated from the frozen
v1.3.7.1 understanding and scoring layers. It:

1. reads the original image (not the Qwen text-masked view);
2. predicts a fixation-style saliency map with TranSalNet_Res;
3. computes interpretable global and OCR-region metrics;
4. saves a grayscale heatmap and an overlay;
5. returns JSON-serialisable evidence without changing any score.

The network structure is an inference-compatible adaptation of the official
MIT-licensed TranSalNet implementation:
https://github.com/LJOVO/TranSalNet

The official pretrained TranSalNet_Res checkpoint is required separately.
"""

from __future__ import annotations

import math
import re
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np


MODEL_NAME = "TranSalNet_Res"
MODULE_VERSION = "transalnet_v1.4.1-candidate"
INPUT_WIDTH = 384
INPUT_HEIGHT = 288
DEFAULT_MODEL_PATH = "models/transalnet/TranSalNet_Res.pth"

_MODEL_CACHE: Dict[Tuple[str, str], Any] = {}
_MODEL_LOCK = threading.Lock()


def _round(value: float) -> float:
    return round(float(value), 4)


def _safe_stem(value: str) -> str:
    text = re.sub(r"[^0-9A-Za-z._-]+", "_", str(value or "").strip())
    return text.strip("._-") or "image"


def _normalize_text(value: Any) -> str:
    return re.sub(r"[^0-9A-Z]+", "", str(value or "").upper())


def _resolve_device(requested: str = "auto") -> str:
    import torch

    value = str(requested or "auto").strip().lower()
    if value == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("请求使用CUDA，但当前PyTorch未检测到可用GPU")
    return value


def _preprocess_image(image: np.ndarray) -> np.ndarray:
    """Match the official 384x288 aspect-preserving padding procedure."""
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("TranSalNet需要三通道BGR图片")

    target_h, target_w = INPUT_HEIGHT, INPUT_WIDTH
    original_h, original_w = image.shape[:2]
    if original_h <= 0 or original_w <= 0:
        raise ValueError("图片尺寸无效")

    # Official code uses np.ones(uint8), so padding value is 1 rather than 255.
    padded = np.ones((target_h, target_w, 3), dtype=np.uint8)
    rows_rate = original_h / target_h
    cols_rate = original_w / target_w

    if rows_rate > cols_rate:
        new_w = max(1, int((original_w * target_h) // original_h))
        new_w = min(new_w, target_w)
        resized = cv2.resize(image, (new_w, target_h), interpolation=cv2.INTER_LINEAR)
        left = (target_w - new_w) // 2
        padded[:, left:left + new_w] = resized
    else:
        new_h = max(1, int((original_h * target_w) // original_w))
        new_h = min(new_h, target_h)
        resized = cv2.resize(image, (target_w, new_h), interpolation=cv2.INTER_LINEAR)
        top = (target_h - new_h) // 2
        padded[top:top + new_h, :] = resized

    return padded


def _restore_to_original(prediction: np.ndarray, original_shape: Sequence[int]) -> np.ndarray:
    """Undo the padding/resizing and return a float32 map in [0, 1]."""
    original_h, original_w = int(original_shape[0]), int(original_shape[1])
    if original_h <= 0 or original_w <= 0:
        raise ValueError("原图尺寸无效")

    pred = np.asarray(prediction, dtype=np.float32)
    pred = np.squeeze(pred)
    if pred.ndim != 2:
        raise ValueError(f"显著性输出应为二维，实际为 {pred.shape}")
    pred = np.clip(pred, 0.0, 1.0)

    pred_h, pred_w = pred.shape
    rows_rate = original_h / pred_h
    cols_rate = original_w / pred_w

    if rows_rate > cols_rate:
        new_w = max(original_w, int((pred_w * original_h) // pred_h))
        resized = cv2.resize(pred, (new_w, original_h), interpolation=cv2.INTER_LINEAR)
        left = max(0, (new_w - original_w) // 2)
        restored = resized[:, left:left + original_w]
    else:
        new_h = max(original_h, int((pred_h * original_w) // pred_w))
        resized = cv2.resize(pred, (original_w, new_h), interpolation=cv2.INTER_LINEAR)
        top = max(0, (new_h - original_h) // 2)
        restored = resized[top:top + original_h, :]

    if restored.shape != (original_h, original_w):
        restored = cv2.resize(restored, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
    return np.clip(restored.astype(np.float32), 0.0, 1.0)


def _extract_state_dict(checkpoint: Any) -> Mapping[str, Any]:
    if isinstance(checkpoint, Mapping):
        for key in ("state_dict", "model_state_dict", "model"):
            nested = checkpoint.get(key)
            if isinstance(nested, Mapping):
                checkpoint = nested
                break
    if not isinstance(checkpoint, Mapping):
        raise TypeError("TranSalNet权重不是有效的state_dict")

    cleaned: Dict[str, Any] = {}
    for key, value in checkpoint.items():
        name = str(key)
        if name.startswith("module."):
            name = name[len("module."):]
        cleaned[name] = value
    return cleaned


def _build_model() -> Any:
    """Build an architecture whose state-dict keys match TranSalNet_Res."""
    import torch
    import torch.nn as nn
    from torchvision import models

    class Attention(nn.Module):
        def __init__(self, config: Mapping[str, Any]):
            super().__init__()
            self.num_attention_heads = int(config["num_heads"])
            self.attention_head_size = int(config["hidden_size"] / self.num_attention_heads)
            self.all_head_size = self.num_attention_heads * self.attention_head_size
            self.query = nn.Linear(config["hidden_size"], self.all_head_size)
            self.key = nn.Linear(config["hidden_size"], self.all_head_size)
            self.value = nn.Linear(config["hidden_size"], self.all_head_size)
            self.out = nn.Linear(self.all_head_size, config["hidden_size"])
            self.attn_dropout = nn.Dropout(config["attention_dropout_rate"])
            self.proj_dropout = nn.Dropout(config["attention_dropout_rate"])
            self.softmax = nn.Softmax(dim=-1)

        def transpose_for_scores(self, x: Any) -> Any:
            new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
            x = x.view(*new_shape)
            return x.permute(0, 2, 1, 3)

        def forward(self, hidden_states: Any) -> Any:
            query_layer = self.transpose_for_scores(self.query(hidden_states))
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            scores = scores / math.sqrt(self.attention_head_size)
            probs = self.attn_dropout(self.softmax(scores))
            context = torch.matmul(probs, value_layer)
            context = context.permute(0, 2, 1, 3).contiguous()
            new_shape = context.size()[:-2] + (self.all_head_size,)
            context = context.view(*new_shape)
            return self.proj_dropout(self.out(context))

    class Mlp(nn.Module):
        def __init__(self, config: Mapping[str, Any]):
            super().__init__()
            self.fc1 = nn.Linear(config["hidden_size"], config["mlp_dim"])
            self.fc2 = nn.Linear(config["mlp_dim"], config["hidden_size"])
            self.act_fn = torch.nn.functional.gelu
            self.dropout = nn.Dropout(config["dropout_rate"])
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.xavier_uniform_(self.fc2.weight)
            nn.init.normal_(self.fc1.bias, std=1e-6)
            nn.init.normal_(self.fc2.bias, std=1e-6)

        def forward(self, x: Any) -> Any:
            x = self.dropout(self.act_fn(self.fc1(x)))
            return self.dropout(self.fc2(x))

    class Block(nn.Module):
        def __init__(self, config: Mapping[str, Any]):
            super().__init__()
            self.flag = config["num_heads"]
            self.hidden_size = config["hidden_size"]
            self.ffn_norm = nn.LayerNorm(config["hidden_size"], eps=1e-6)
            self.ffn = Mlp(config)
            self.attn = Attention(config)
            self.attention_norm = nn.LayerNorm(config["hidden_size"], eps=1e-6)

        def forward(self, x: Any) -> Any:
            h = x
            x = self.attn(self.attention_norm(x)) + h
            h = x
            return self.ffn(self.ffn_norm(x)) + h

    class Encoder(nn.Module):
        def __init__(self, config: Mapping[str, Any]):
            super().__init__()
            import copy

            self.layer = nn.ModuleList()
            self.encoder_norm = nn.LayerNorm(config["hidden_size"], eps=1e-6)
            for _ in range(config["num_layers"]):
                self.layer.append(copy.deepcopy(Block(config)))

        def forward(self, hidden_states: Any) -> Any:
            for layer_block in self.layer:
                hidden_states = layer_block(hidden_states)
            return self.encoder_norm(hidden_states)

    class TransEncoder(nn.Module):
        def __init__(self, in_channels: int, spatial_size: int, config: Mapping[str, Any]):
            super().__init__()
            self.patch_embeddings = nn.Conv2d(in_channels, config["hidden_size"], kernel_size=1, stride=1)
            self.position_embeddings = nn.Parameter(torch.zeros(1, spatial_size, config["hidden_size"]))
            self.transformer_encoder = Encoder(config)

        def forward(self, x: Any) -> Any:
            height, width = x.shape[2], x.shape[3]
            x = self.patch_embeddings(x).flatten(2).transpose(-1, -2)
            x = self.transformer_encoder(x + self.position_embeddings)
            batch, _, hidden = x.shape
            x = x.permute(0, 2, 1)
            return x.contiguous().view(batch, hidden, height, width)

    cfg1 = {
        "hidden_size": 768, "mlp_dim": 3072, "num_heads": 12,
        "num_layers": 2, "attention_dropout_rate": 0.0, "dropout_rate": 0.0,
    }
    cfg2 = dict(cfg1)
    cfg3 = {
        "hidden_size": 512, "mlp_dim": 2048, "num_heads": 8,
        "num_layers": 2, "attention_dropout_rate": 0.0, "dropout_rate": 0.0,
    }

    class _Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            try:
                backbone = models.resnet50(weights=None)
            except TypeError:
                backbone = models.resnet50(pretrained=False)
            self.encoder = nn.ModuleList(list(backbone.children())[:8]).eval()

        def forward(self, x: Any) -> List[Any]:
            outputs: List[Any] = []
            for index, layer in enumerate(self.encoder):
                x = layer(x)
                if index in {5, 6, 7}:
                    outputs.append(x)
            return outputs

    class _Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(768, 768, 3, 1, 1)
            self.conv2 = nn.Conv2d(768, 512, 3, 1, 1)
            self.conv3 = nn.Conv2d(512, 256, 3, 1, 1)
            self.conv4 = nn.Conv2d(256, 128, 3, 1, 1)
            self.conv5 = nn.Conv2d(128, 64, 3, 1, 1)
            self.conv6 = nn.Conv2d(64, 32, 3, 1, 1)
            self.conv7 = nn.Conv2d(32, 1, 3, 1, 1)
            self.batchnorm1 = nn.BatchNorm2d(768)
            self.batchnorm2 = nn.BatchNorm2d(512)
            self.batchnorm3 = nn.BatchNorm2d(256)
            self.batchnorm4 = nn.BatchNorm2d(128)
            self.batchnorm5 = nn.BatchNorm2d(64)
            self.batchnorm6 = nn.BatchNorm2d(32)
            self.TransEncoder1 = TransEncoder(2048, 9 * 12, cfg1)
            self.TransEncoder2 = TransEncoder(1024, 18 * 24, cfg2)
            self.TransEncoder3 = TransEncoder(512, 36 * 48, cfg3)
            self.add = torch.add
            self.relu = nn.ReLU(True)
            self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
            self.sigmoid = nn.Sigmoid()

        def forward(self, features: Sequence[Any]) -> Any:
            x3, x4, x5 = features
            x5 = self.relu(self.batchnorm1(self.conv1(self.TransEncoder1(x5))))
            x5 = self.upsample(x5)
            x4 = self.relu(x5 * self.TransEncoder2(x4))
            x4 = self.upsample(self.relu(self.batchnorm2(self.conv2(x4))))
            x3 = self.relu(x4 * self.TransEncoder3(x3))
            x3 = self.upsample(self.relu(self.batchnorm3(self.conv3(x3))))
            x2 = self.upsample(self.relu(self.batchnorm4(self.conv4(x3))))
            x2 = self.relu(self.batchnorm5(self.conv5(x2)))
            x1 = self.upsample(x2)
            x1 = self.relu(self.batchnorm6(self.conv6(x1)))
            return self.sigmoid(self.conv7(x1))

    class TranSalNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = _Encoder()
            self.decoder = _Decoder()

        def forward(self, x: Any) -> Any:
            return self.decoder(self.encoder(x))

    return TranSalNet()


def _get_model(model_path: str, device: str) -> Any:
    import torch

    resolved = str(Path(model_path).expanduser().resolve())
    cache_key = (resolved, device)
    with _MODEL_LOCK:
        if cache_key in _MODEL_CACHE:
            return _MODEL_CACHE[cache_key]

        model = _build_model()
        try:
            checkpoint = torch.load(resolved, map_location="cpu", weights_only=True)
        except TypeError:
            checkpoint = torch.load(resolved, map_location="cpu")
        state_dict = _extract_state_dict(checkpoint)
        model.load_state_dict(state_dict, strict=True)
        model = model.to(torch.device(device))
        model.eval()
        _MODEL_CACHE[cache_key] = model
        return model


def _box_to_rect(box: Any, width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    if box is None:
        return None
    try:
        array = np.asarray(box, dtype=np.float32)
    except Exception:
        return None

    if array.size < 4:
        return None
    if array.ndim == 1 and array.size >= 4:
        x1, y1, x2, y2 = array[:4].tolist()
    else:
        points = array.reshape(-1, 2)
        x1, y1 = points.min(axis=0).tolist()
        x2, y2 = points.max(axis=0).tolist()

    left = max(0, min(width, int(math.floor(min(x1, x2)))))
    right = max(0, min(width, int(math.ceil(max(x1, x2)))))
    top = max(0, min(height, int(math.floor(min(y1, y2)))))
    bottom = max(0, min(height, int(math.ceil(max(y1, y2)))))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _build_mask(
    text_items: Iterable[Mapping[str, Any]],
    width: int,
    height: int,
    target_texts: Optional[Iterable[str]] = None,
) -> Tuple[np.ndarray, int]:
    mask = np.zeros((height, width), dtype=bool)
    normalized_targets = {_normalize_text(value) for value in (target_texts or []) if _normalize_text(value)}
    matched = 0

    for item in text_items or []:
        text = _normalize_text(item.get("text"))
        if normalized_targets:
            is_match = any(
                text == target or (len(target) >= 4 and target in text) or (len(text) >= 4 and text in target)
                for target in normalized_targets
            )
            if not is_match:
                continue
        rect = _box_to_rect(item.get("box"), width, height)
        if rect is None:
            continue
        left, top, right, bottom = rect
        mask[top:bottom, left:right] = True
        matched += 1
    return mask, matched


def _region_statistics(saliency: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
    if mask.shape != saliency.shape or not np.any(mask):
        return {
            "box_count": 0,
            "area_ratio": 0.0,
            "saliency_mass_ratio": 0.0,
            "mean_saliency": 0.0,
            "saliency_gain": 0.0,
        }

    values = np.clip(saliency.astype(np.float64), 0.0, None)
    total_mass = float(values.sum())
    global_mean = float(values.mean())
    region_mean = float(values[mask].mean())
    mass_ratio = float(values[mask].sum() / total_mass) if total_mass > 0 else 0.0
    gain = float(region_mean / global_mean) if global_mean > 1e-12 else 0.0
    return {
        "area_ratio": _round(float(mask.mean())),
        "saliency_mass_ratio": _round(mass_ratio),
        "mean_saliency": _round(region_mean),
        "saliency_gain": _round(gain),
    }


def compute_saliency_metrics(saliency: np.ndarray) -> Dict[str, float]:
    values = np.clip(np.asarray(saliency, dtype=np.float64), 0.0, None)
    if values.ndim != 2 or values.size == 0:
        raise ValueError("显著性图必须是非空二维数组")

    total = float(values.sum())
    if total <= 1e-12:
        probabilities = np.full(values.size, 1.0 / values.size, dtype=np.float64)
    else:
        probabilities = (values / total).ravel()

    nonzero = probabilities[probabilities > 0]
    entropy = -float(np.sum(nonzero * np.log(nonzero)))
    normalized_entropy = entropy / math.log(values.size) if values.size > 1 else 0.0

    sorted_mass = np.sort(probabilities)[::-1]
    top10_count = max(1, int(math.ceil(values.size * 0.10)))
    top10_mass = float(sorted_mass[:top10_count].sum())
    half_mass_index = int(np.searchsorted(np.cumsum(sorted_mass), 0.5, side="left")) + 1
    half_mass_area_ratio = half_mass_index / values.size

    height, width = values.shape
    y1, y2 = int(height * 0.25), int(height * 0.75)
    x1, x2 = int(width * 0.25), int(width * 0.75)
    center_mass = float(values[y1:y2, x1:x2].sum() / total) if total > 0 else 0.25

    peak_index = int(np.argmax(values))
    peak_y, peak_x = np.unravel_index(peak_index, values.shape)

    return {
        "attention_entropy": _round(normalized_entropy),
        "attention_concentration": _round(top10_mass),
        "top10_saliency_mass_ratio": _round(top10_mass),
        "half_mass_area_ratio": _round(half_mass_area_ratio),
        "center_saliency_ratio": _round(center_mass),
        "mean_saliency": _round(float(values.mean())),
        "peak_saliency": _round(float(values.max())),
        "peak_x_ratio": _round(float(peak_x / max(width - 1, 1))),
        "peak_y_ratio": _round(float(peak_y / max(height - 1, 1))),
    }


def _save_visualizations(
    image: np.ndarray,
    saliency: np.ndarray,
    output_dir: str,
    output_stem: str,
) -> Tuple[str, str]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    stem = _safe_stem(output_stem)
    heatmap_path = target / f"{stem}_transalnet_heatmap.png"
    overlay_path = target / f"{stem}_transalnet_overlay.jpg"

    heatmap_u8 = np.clip(saliency * 255.0, 0, 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.58, heatmap_color, 0.42, 0.0)

    if not cv2.imwrite(str(heatmap_path), heatmap_u8):
        raise OSError(f"无法保存显著性图：{heatmap_path}")
    if not cv2.imwrite(str(overlay_path), overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 94]):
        raise OSError(f"无法保存显著性叠加图：{overlay_path}")
    return str(heatmap_path), str(overlay_path)


def analyze_saliency(
    image_path: str,
    ocr_result: Optional[Mapping[str, Any]] = None,
    output_dir: str = "outputs/saliency",
    output_stem: Optional[str] = None,
    model_path: str = DEFAULT_MODEL_PATH,
    device: str = "auto",
    save_visualizations: bool = True,
    strict: bool = False,
) -> Dict[str, Any]:
    """Run TranSalNet independently and return interpretable evidence.

    The returned result is never consumed by the v1.3.7.1 score engine. When
    ``strict`` is False, deployment errors are returned as structured status
    values so the original pipeline remains available.
    """
    try:
        import torch

        image_file = Path(image_path)
        checkpoint_file = Path(model_path).expanduser()
        if not image_file.exists():
            raise FileNotFoundError(f"找不到图片：{image_path}")
        if not checkpoint_file.exists():
            return {
                "status": "model_missing",
                "enabled": True,
                "experimental": True,
                "module_version": MODULE_VERSION,
                "model": MODEL_NAME,
                "checkpoint": str(checkpoint_file),
                "input_view": "original",
                "scoring_integration": False,
                "error": f"找不到TranSalNet权重：{checkpoint_file}",
            }

        image = cv2.imread(str(image_file), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"OpenCV无法读取图片：{image_path}")
        original_h, original_w = image.shape[:2]
        resolved_device = _resolve_device(device)
        model = _get_model(str(checkpoint_file), resolved_device)

        preprocessed = _preprocess_image(image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(np.transpose(preprocessed, (2, 0, 1))).unsqueeze(0)
        tensor = tensor.to(torch.device(resolved_device), dtype=torch.float32)

        with torch.inference_mode():
            prediction = model(tensor)
        saliency_small = prediction.detach().float().cpu().numpy().squeeze()
        saliency = _restore_to_original(saliency_small, image.shape)

        metrics = compute_saliency_metrics(saliency)
        ocr = dict(ocr_result or {})
        text_items = ocr.get("text_items", []) or []
        text_mask, text_count = _build_mask(text_items, original_w, original_h)
        cta_mask, cta_count = _build_mask(
            text_items, original_w, original_h, ocr.get("cta_text", []) or []
        )
        brand_mask, brand_count = _build_mask(
            text_items, original_w, original_h, ocr.get("possible_brand_words", []) or []
        )

        text_stats = _region_statistics(saliency, text_mask)
        text_stats["box_count"] = text_count
        cta_stats = _region_statistics(saliency, cta_mask)
        cta_stats["box_count"] = cta_count
        brand_stats = _region_statistics(saliency, brand_mask)
        brand_stats["box_count"] = brand_count

        heatmap_path = ""
        overlay_path = ""
        if save_visualizations:
            heatmap_path, overlay_path = _save_visualizations(
                image=image,
                saliency=saliency,
                output_dir=output_dir,
                output_stem=output_stem or image_file.stem,
            )

        return {
            "status": "ok",
            "enabled": True,
            "experimental": True,
            "module_version": MODULE_VERSION,
            "model": MODEL_NAME,
            "checkpoint": str(checkpoint_file),
            "device": resolved_device,
            "input_view": "original",
            "input_size": {"width": INPUT_WIDTH, "height": INPUT_HEIGHT},
            "original_size": {"width": original_w, "height": original_h},
            "metrics": metrics,
            "region_metrics": {
                "all_text": text_stats,
                "cta": cta_stats,
                "brand_candidates": brand_stats,
            },
            "heatmap_path": heatmap_path,
            "overlay_path": overlay_path,
            "scoring_integration": False,
            "note": "实验性独立证据；未进入v1.3.7.1理解层或五维评分。",
            "error": None,
        }
    except Exception as exc:
        if strict:
            raise
        return {
            "status": "failed",
            "enabled": True,
            "experimental": True,
            "module_version": MODULE_VERSION,
            "model": MODEL_NAME,
            "checkpoint": str(model_path),
            "input_view": "original",
            "scoring_integration": False,
            "error": f"TranSalNet failed: {type(exc).__name__}: {exc}",
        }
