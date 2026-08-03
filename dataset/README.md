# Dataset

该目录保存数据卡、公开样本、划分文件和校验信息，不提交完整私有工作区、临时图片或未授权商业素材。

推荐结构：

```text
dataset/
├── AIGC_images/
│   └── sample/
├── splits/
│   ├── dev.txt
│   ├── regression.txt
│   ├── holdout.txt
│   └── external.txt
├── prompts.jsonl
└── dataset_checksums.sha256
```

完整数据集发布前需完成版权、隐私、人工标注和哈希检查。详见 `docs/DATASET.md`。
