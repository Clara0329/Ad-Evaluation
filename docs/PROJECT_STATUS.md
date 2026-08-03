# 项目当前状态

## 已完成

1. PaddleOCR 中英文广告文字提取与文本状态判断。
2. OpenCV 亮度、对比度、饱和度、清晰度、边缘与布局特征。
3. Qwen2-VL 遮罩文字后的可见语义提取。
4. 场景分类、文本角色识别、主体对齐和证据富化。
5. 多源 Evidence Validator 与来源可靠度。
6. 七类场景路由的五维规则评分。
7. TranSalNet 显著性热力图与解释指标。
8. 36 张显著性人工验证、288 张批处理、A/B 分析和 5 折折外验证。
9. TranSalNet 注意力候选并行输出，正式评分保持冻结。

## 当前版本

| 部分 | 版本 | 状态 |
|---|---|---|
| 正式理解与评分主干 | `mvp_ocr_qwen_validator_score_v1.3.7.1` | 冻结 |
| 正式评分引擎 | `scene_category_source_aware_v3.1` | 工程初值 |
| TranSalNet 推理适配 | `transalnet_v1.4.1-candidate` | 实验性 |
| 注意力候选输出 | `transalnet_v1.4.3-attention-parallel` | 并行、未接入正式评分 |

## 当前数据

- ChatGPT：108 张
- 豆包：72 张
- Midjourney：108 张
- Stable Diffusion：0 张
- 已完成 TranSalNet 内部实验：288 张

## 下一阶段唯一主线

建立独立外部验证集并进行人工五维评分。优先比较：

1. 人工评分者一致性；
2. 系统五维分与人工均值的 Spearman 相关；
3. MAE/RMSE；
4. 同 Prompt 四模型排序一致性；
5. 正式注意力与候选注意力的外部表现。

在外部验证完成前，不继续针对原 288 张调参，也不将候选注意力覆盖正式评分。
