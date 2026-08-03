# 系统方法

## 1. 设计原则

本项目采用“理解与评分分离”的可解释架构。多模态模型只负责提取可见语义，不直接输出心理分数。评分引擎只消费经过校验的结构化证据。

## 2. OCR 硬事实层

`modules/ocr_module.py` 使用 PaddleOCR，输出：

- OCR 文字及置信度、位置框；
- 价格、促销、CTA、活动日期时间和地点候选；
- 品牌候选；
- `absent / readable / partially_readable / garbled` 文字状态；
- 可疑乱码和文本角色分析。

价格、活动时间、地点、CTA 等硬事实优先由 OCR 及规则确认，不能仅凭 Qwen 推断进入最终证据。

## 3. 传统视觉特征层

`modules/visual_module.py` 基于 OpenCV 提取：

- brightness、contrast、saturation；
- sharpness、edge_density、border_complexity；
- layout_complexity；
- center_focus 与兼容字段 saliency_center；
- dominant_colors；
- too_dark、too_bright、too_cluttered、low_sharpness。

这些指标承担稳定、可重复的低层视觉描述，不由多模态模型替代。

## 4. Qwen2-VL 可见语义层

`modules/qwen_module.py` 的核心约束：

- OCR 读取原图；Qwen 读取文字遮罩图；
- 禁止 Qwen 猜测品牌、价格、优惠、CTA、网址和型号；
- 只提取主体、产品或内容类型、场景、构图、色彩、光影、情绪、注意元素和记忆元素；
- 解析失败时使用本地保守兜底，不重复进行高成本多模态推理。

默认模型为 `Qwen/Qwen2-VL-2B-Instruct`，项目可通过 `QWEN_MODEL_PATH` 指向本地快照。

## 5. 证据对齐与校验

相关模块：

- `scene_classifier.py`：结合 OCR 和 Qwen 修正场景；
- `text_role_classifier.py`：文本角色、CTA 和品牌语言评估；
- `subject_resolver.py`：核心主体候选解析；
- `subject_aligner.py`：主体与场景对齐；
- `evidence_enricher.py`：补齐明确 OCR 事实；
- `evidence_validator.py`：来源记录、去重、冲突处理和幻觉门控。

证据来源权重：

| 来源 | 工程可靠度 |
|---|---:|
| OCR | 1.00 |
| Qwen | 0.88 |
| 视觉特征 | 0.82 |
| SAM 兼容分支 | 0.82 |
| 规则推断 | 0.62 |

## 6. 场景化评分

`modules/score_engine.py` 支持：

- 电商商品广告
- 品牌广告
- 活动宣传海报
- 短视频封面
- 教育校园宣传
- 公益宣传
- 旅游宣传
- 其他（通用评分）

每个场景使用不同的证据组合与权重，但输出统一五维结构和子规则证据。

## 7. TranSalNet 独立显著性分支

`modules/transalnet_module.py`读取原图，输出：

- attention_entropy
- attention_concentration
- half_mass_area_ratio
- center_saliency_ratio
- peak 位置
- OCR 区域显著性
- 热力图和叠加图

当前只有 `center_saliency_ratio`通过初步人工中心判断验证。候选注意力通过 `modules/attention_candidate_module.py`产生，但始终保持：

```json
{
  "candidate_scope": "attention_only",
  "scoring_integration": false
}
```

## 8. 可解释输出

正式 JSON 同时保留：

- 原始 OCR 与视觉特征；
- Qwen 原始结构化结果；
- 校验后的最终证据；
- 证据来源与校验备注；
- 场景评分配置；
- 五维子规则依据；
- 显著性热力图路径；
- 候选注意力与正式注意力的差异。
