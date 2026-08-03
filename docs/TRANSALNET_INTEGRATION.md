# TranSalNet 注意力并行候选

## 定位

TranSalNet 是独立视觉显著性分支，不替代 OCR、OpenCV、Qwen 或 Validator。

```json
{
  "experimental": true,
  "candidate_scope": "attention_only",
  "scoring_integration": false
}
```

## 输出指标

- `attention_entropy`：显著性分布熵；
- `attention_concentration`：最高显著区域承载质量；
- `half_mass_area_ratio`：承载50%显著质量所需面积；
- `center_saliency_ratio`：中央区域显著质量比例；
- OCR、CTA 和品牌候选区域统计；
- 热力图与原图叠加图。

当前只有 `center_saliency_ratio`通过初步人工中心判断验证。CTA 和品牌区域指标暂不进入正式评分。

## 候选校准

`configs/attention_calibration_v143.json`保存 288 张训练样本的全局和场景分位数映射。场景样本少于20时使用全局回退。

候选模块只重新计算注意力分，其余四维保持正式结果：

```text
正式 persuasion = 候选 persuasion
正式 arousal    = 候选 arousal
正式 trust      = 候选 trust
正式 memory     = 候选 memory
候选 attention  = 使用校准后的中心显著证据重算
```

## 使用

```bash
python main.py \
  --image image.jpg \
  --output outputs/transalnet \
  --use-qwen \
  --use-transalnet \
  --transalnet-model models/transalnet/TranSalNet_Res.pth \
  --transalnet-device auto
```

## 研究边界

显著性预测不等同于眼动实验。候选注意力只有在独立外部人工验证显著优于正式注意力时，才可考虑进入正式评分。
