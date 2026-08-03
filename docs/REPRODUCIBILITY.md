# 复现说明

## 1. 推荐环境

- Linux
- Python 3.10
- Transformers 4.51.3
- 支持 CUDA 的 PyTorch（Qwen 与 TranSalNet 推荐 GPU）
- PaddleOCR 3.x

先按服务器 CUDA 版本安装匹配的 PyTorch 和 PaddlePaddle，再安装：

```bash
pip install -r requirements.txt
```

## 2. 模型目录

```text
models/
├── Qwen2-VL-2B-Instruct/            # 或设置QWEN_MODEL_PATH
└── transalnet/
    └── TranSalNet_Res.pth
```

模型权重不随仓库发布。

## 3. 单图命令

```bash
python main.py \
  --image /path/to/image.png \
  --output outputs/run \
  --use-qwen \
  --use-transalnet \
  --transalnet-model models/transalnet/TranSalNet_Res.pth \
  --transalnet-device auto
```

## 4. 批处理列表

`images.txt` 每行一个路径，可使用 `#` 注释：

```text
/path/to/A001.png
/path/to/A002.png
```

执行：

```bash
python main.py \
  --list images.txt \
  --output outputs/batch \
  --use-qwen \
  --use-transalnet \
  --skip-existing
```

## 5. 静态检查

```bash
python tools/check_repository.py
```

该检查不加载 PaddleOCR、Qwen 或 TranSalNet 权重，只验证 Python 语法、关键文件、校准配置和纯规则评分接口。

## 6. 结果复现

每个结果 JSON 包含：

- 输入文件 SHA256；
- 流水线版本；
- 创建时间和模块耗时；
- OCR、视觉、Qwen、Validator 和评分结果；
- 显著性模型状态与候选评分状态。

使用固定原图和固定代码环境，应得到一致或接近一致的确定性特征；Qwen/PaddleOCR 在不同硬件和软件版本下可能产生轻微差异，因此论文应公开环境、代码标签和原始结果 JSON。

## 7. 当前服务器参考

项目历史开发环境使用 Python 3.10、Transformers 4.51.3 和 Qwen2-VL-2B-Instruct。正式论文发布时应再导出精确环境快照：

```bash
pip freeze > environment/pip-freeze.txt
conda env export --no-builds > environment/conda-environment.yml
```
