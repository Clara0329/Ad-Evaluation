# 视觉内容心理效应量化测评系统

> AIGC 视觉内容心理影响力测评基准与工具开发  
> 上海市大学生创新创业训练计划 · 创新训练项目  
> 人工智能 × 应用心理学 × 广告传播

本项目面向广告海报、电商主图、短视频封面、校园与公益宣传、旅游推广等视觉内容，构建一套**可解释、可追溯、可复现**的五维心理效应量化框架。系统不让多模态大模型直接给最终分数，而是将 OCR 硬事实、传统视觉特征、Qwen2-VL 可见语义、证据校验和场景化规则评分拆分为独立模块，再输出每项分数的证据依据。

当前正式主干版本：`mvp_ocr_qwen_validator_score_v1.3.7.1`  
当前显著性并行候选版本：`transalnet_v1.4.3-attention-parallel`

## 项目定位

本仓库同时服务于两类目标：

- **大创展示版**：突出研究问题、系统结构、创新点、阶段性成果和后续计划；
- **论文代码版**：保留可复现代码、实验协议、评分规则、数据组织和验证结果。

当前系统输出五个维度：

| 维度 | 含义 |
|---|---|
| `persuasion` | 消费说服度：是否促进购买、点击、报名或进一步行动 |
| `arousal` | 情绪唤醒度：是否引起兴奋、好奇、紧迫或其他明显情绪反应 |
| `trust` | 信任度：品牌、信息、主体和表达是否显得可靠 |
| `attention` | 注意力吸引度：是否能快速形成清晰视觉焦点 |
| `memory` | 记忆留存度：主体、品牌、口号、数字或视觉符号是否容易被记住 |

> 当前分数是理论与文献启发的工程规则初值，正式心理学效度仍需通过独立人工实验标定。本仓库不会把工程分数表述为已经得到的真实心理测量值。

## 当前架构

```text
原始图片
  ├─ PaddleOCR：文字、价格、促销、CTA、活动时间与地点候选
  ├─ OpenCV：亮度、对比度、饱和度、清晰度、边缘与布局复杂度
  ├─ Qwen2-VL：遮罩文字后的可见主体、场景、构图、情绪和视觉证据
  └─ TranSalNet（实验分支）：视觉显著性热力图和中心显著度
            ↓
场景分类 + 文本角色判断 + 主体对齐
            ↓
Evidence Validator：多源证据合并、去重、来源标记和幻觉门控
            ↓
场景感知五维规则评分
            ↓
JSON：正式评分 + 子规则证据 + 显著性并行候选
```

YOLO + SAM 仅保留为兼容性实验模块，不属于当前默认主路线。其在广告海报、文字密集封面和复杂商业构图中的泛化能力不足，因此正式流程采用“硬事实 OCR + 可见语义理解 + 证据校验”的结构。

## 关键创新

1. **多源证据而非黑盒总分**：Qwen2-VL 只提取语义证据，不直接评分；硬事实优先由 OCR 确认。
2. **场景感知评分**：针对电商商品广告、品牌广告、活动海报、短视频封面、教育校园、公益、旅游等场景使用不同规则配置。
3. **证据来源可靠度**：OCR、Qwen、视觉规则和规则推断分别记录来源与可靠度，降低无依据品牌、价格、CTA 和活动信息进入评分的概率。
4. **显著性独立分支**：TranSalNet 生成热力图和注意力候选分，但 `scoring_integration=false`，不污染冻结的正式评分。
5. **可复现实验设计**：固定 Prompt、模型、图片、数据划分、文件哈希和版本输出，支持样本复现、方法复现与结论复现。

## 阶段性结果

### TranSalNet 人工热点验证（36 张）

- 第一热点完全一致率：83.3%
- 第一热点一致或部分一致率：91.7%
- 广告主体完全覆盖率：88.9%
- 广告主体完全或部分覆盖率：100%
- 无意义热点：0 张
- `center_saliency_ratio` 与人工中心判断：Spearman `ρ = 0.5015`，`p = 0.0018`

因此当前只允许 `center_saliency_ratio`进入候选校准；`attention_entropy`、`attention_concentration`和`half_mass_area_ratio`仅作为解释性输出。

### 288 张内部实验

- ChatGPT：108 张
- 豆包：72 张
- Midjourney：108 张
- TranSalNet 成功：288/288
- 候选字段有效：288/288
- 正式评分接入：0 张（全部保持 `false`）

直接用 TranSalNet 原始候选替换旧中心特征会造成系统性降分，因此该方案被否定。随后采用场景分位数映射，并完成 5 折场景分层折外验证：

- 注意力变化均值：-0.0764
- 注意力变化中位数：0
- 注意力绝对变化 ≥ 10：0 张
- 总分变化均值：-0.0153
- 总分绝对变化 ≥ 2：0 张

该结果证明候选分数尺度稳定，但**尚未证明候选注意力比正式注意力更符合人类评价**。下一阶段需使用全新图片进行外部人工验证。

## 快速开始

### 1. 环境

推荐：Python 3.10、PyTorch、Transformers 4.51.3。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

GPU 版本 PaddlePaddle 和 PyTorch 应根据服务器 CUDA 环境单独安装。详见 [复现说明](docs/REPRODUCIBILITY.md)。

### 2. 单张图片

基础流程：

```bash
python main.py \
  --image examples/sample.jpg \
  --output outputs/demo \
  --use-qwen
```

启用 TranSalNet 并行解释：

```bash
python main.py \
  --image examples/sample.jpg \
  --output outputs/demo \
  --use-qwen \
  --use-transalnet \
  --transalnet-model models/transalnet/TranSalNet_Res.pth \
  --transalnet-device auto
```

### 3. 批处理

```bash
python main.py \
  --list examples/images.txt \
  --output outputs/batch \
  --use-qwen \
  --use-transalnet \
  --skip-existing
```

## 输出结构

```json
{
  "pipeline_version": "mvp_ocr_qwen_validator_score_v1.3.7.1",
  "ocr": {},
  "visual_features": {},
  "qwen_analysis": {},
  "validated_evidence": {},
  "saliency_analysis": {
    "scoring_integration": false
  },
  "attention_candidate": {
    "candidate_scope": "attention_only",
    "scoring_integration": false
  },
  "score_result": {
    "scores": {
      "persuasion": 0,
      "arousal": 0,
      "trust": 0,
      "attention": 0,
      "memory": 0,
      "total": 0
    },
    "subrule_evidence": {}
  }
}
```

## 仓库目录

```text
Ad-Evaluation/
├── main.py
├── modules/                  # OCR、视觉、Qwen、校验、评分、显著性模块
├── configs/                  # 注意力候选校准配置
├── tools/                    # 批量审计、回放、A/B与交叉验证脚本
├── tests/                    # 不加载模型的基础测试
├── docs/                     # 方法、评分、实验、数据集与复现文档
├── dataset/                  # 数据说明、划分和公开样本入口
├── experiments/              # 实验协议和结果组织规范
├── models/                   # 模型放置说明，不提交权重
├── frontend/                 # 后续可视化系统占位，不虚构已完成功能
└── examples/                 # 输入列表与结果格式示例
```

## 文档

- [项目当前状态](docs/PROJECT_STATUS.md)
- [系统方法](docs/METHOD.md)
- [五维评分规则](docs/SCORING_RULES.md)
- [阶段性实验](docs/EXPERIMENTS.md)
- [数据集与生成协议](docs/DATASET.md)
- [复现说明](docs/REPRODUCIBILITY.md)
- [TranSalNet 并行候选](docs/TRANSALNET_INTEGRATION.md)

## 当前边界

- 不把显著性预测等同于真实眼动实验；
- 不把工程规则分数等同于已经标定的心理测量值；
- 不根据原 288 张继续调参；
- CTA 和品牌区域显著性暂不进入评分；
- Stable Diffusion 数据仍需补充；
- 论文最终结论需等待独立人工评分和外部验证集。

## 团队

- 项目负责人：魏渤函
- 项目成员：赵钰、罗祥悦
- 指导教师：范自柱
- 单位：上海电力大学人工智能学部

## 引用

研究使用本仓库时，可参考 [`CITATION.cff`](CITATION.cff)。TranSalNet 原始方法与代码版权归其作者所有，预训练权重不随本仓库发布。
