# YOLOv11 鼠类检测 → K230 部署

YOLOv11 目标检测模型训练（Colab）+ nncase kmodel 转换（本地），最终部署到 K230 开发板。

## 文件说明

| 文件 | 用途 |
|------|------|
| `yolov11_train.ipynb` | Colab 训练 + 导出 ONNX |
| `onnx2kmodel.py` | 本地 ONNX → kmodel 转换（含量化质量报告） |
| `exp.onnx` | 训练导出的 ONNX 模型 |
| `images/` | PTQ 校准图片（无需标注） |

## 一、Colab 训练

在 Google Colab 中打开 `yolov11_train.ipynb`，按顺序执行：

1. **安装依赖** — `ultralytics` + `roboflow`
2. **下载数据集** — 通过 Roboflow API（替换为自己的 api_key / workspace / project）
3. **训练** — `yolo11n.pt`，imgsz=320，epochs=50
4. **导出 ONNX** — `model.export(format="onnx", imgsz=(320, 320))`

训练完成后，将 `best.onnx`（重命名为 `exp.onnx`）和部分数据集图片下载到本地。

## 二、本地 kmodel 转换

### 环境要求

```
Python 3.10
nncase==2.9.0  nncase-kpu==2.9.0  （K230 固件要求，勿改版本）
onnx  onnxsim  onnxruntime  Pillow  numpy
```

```bash
pip install nncase==2.9.0 nncase-kpu==2.9.0
pip install onnx onnxsim onnxruntime Pillow numpy
```

### 转换

```bash
# 默认方案（全 uint8，最小最快）
python onnx2kmodel.py

# 高精度方案（int16 激活，精度最高）
python onnx2kmodel.py -p 5

# 全部 6 种方案一次转完，输出对比表
python onnx2kmodel.py -a

# 查看帮助
python onnx2kmodel.py -h
```

转换完成后自动输出**逐层量化质量报告**，以最终输出层余弦相似度 >= 0.99 为达标。

### 量化方案选择

| 方案 | 激活/权重 | 体积 | 相似度 | 特点 |
|------|----------|------|--------|------|
| 0 | uint8/uint8 | 2.9 MB | 0.999 | 最小最快，KPU 满速 |
| 5 | int16/uint8 | 3.1 MB | 1.000 | 精度最高，推理稍慢 |

- **实时视频** → 方案 0（全 uint8，跑满 KPU）
- **离线/精度优先** → 方案 5

### 校准图片

从 `images/` 取前 N 张（默认 10），建议保留 **30 张**有代表性的图片，覆盖不同光照/角度/背景。无需标注文件，转换时只读像素。

## 三、部署

将生成的 `exp.kmodel` 拷贝到 K230 SD 卡，配合 MicroPython 推理脚本运行。参考 K230 官方文档的后处理与显示代码。
