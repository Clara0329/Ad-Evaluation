# Models

模型权重不提交到 GitHub。

需要的主要模型：

1. Qwen2-VL-2B-Instruct：放置于 `models/Qwen2-VL-2B-Instruct/`，或设置环境变量 `QWEN_MODEL_PATH`；
2. TranSalNet_Res：放置于 `models/transalnet/TranSalNet_Res.pth`。

`modules/sam_module.py`是历史兼容分支，需要 YOLOv8n 与 SAM ViT-B 权重，但不属于默认系统。
