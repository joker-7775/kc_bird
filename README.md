# 🐦 CUB 鸟类多模态分类

基于 **Caltech-UCSD Birds-200-2011 (CUB)** 数据集，结合 **ResNet-18（图像）** + **BERT（文本）** 的多模态鸟类分类系统。

## 📌 项目概述

本项目对比了三种分类方案：

| 方案 | 模态 | 核心思路 |
|---|---|---|
| **Text-Only** | 仅文本 | 用 BERT 编码鸟类描述文本，直接分类 |
| **Image-Only** | 仅图像 | 用 ResNet-18 提取图像特征，直接分类 |
| **Late Fusion** | 图像 + 文本 | 分别编码后拼接特征，融合分类 |

## 🏗️ 项目结构

```
kc_bird/
├── image_encoder.py          # 图像编码器：ResNet-18 + 投影层
├── text_encoder.py           # 文本编码器：BERT + Mean Pooling + 投影层
├── singleModal_TextOnly.py   # 单模态基线：仅用文本分类
├── multimodal_latefusion.py  # 多模态融合：图像 + 文本 Late Fusion
├── convert_cub_json.py       # 数据预处理：将 CUB 图片/文本转为标注 JSON
├── split_dataset.py          # 数据集划分：8:1:1 切分 train / val / test
├── datasets/                 # [运行时生成] 标注 JSON 存放目录
│   ├── annotations_train.json
│   ├── annotations_val.json
│   └── annotations_test.json
└── CUB_200_subset/           # [需自行准备] CUB 子集数据
    ├── images/<类别>/<图片>.jpg
    └── text/<类别>/<图片>.txt
```

## 🧩 模型架构

### 图像编码器（ImageEncoder）

```
输入 [B, 3, 224, 224]
    ↓
ResNet-18 (去掉分类头)    → [B, 512, 7, 7]
    ↓
AdaptiveAvgPool2d(1,1)   → [B, 512, 1, 1]
    ↓
Flatten                  → [B, 512]
    ↓
Linear(512 → 512)        → [B, 512]  （统一特征维度）
```

### 文本编码器（TextEncoder）

```
输入: input_ids [B, L] + attention_mask [B, L]
    ↓
BERT (bert-base-uncased) → last_hidden_state [B, L, 768]
    ↓
Mean Pooling (MASK 感知) → [B, 768]
    ↓
Linear(768 → 512)        → [B, 512]  （统一特征维度）
```

### Late Fusion 多模态融合

```
  ImageEncoder [B,512] ──┐
                         ├──→ Concat [B,1024] ──→ MLP Classifier → logits [B, num_classes]
  TextEncoder  [B,512] ──┘
```

融合分类器结构：`Linear(1024, 512) → ReLU → Dropout(0.2) → Linear(512, num_classes)`

## 🛠️ 环境准备

### 1. 依赖安装

```bash
pip install torch torchvision transformers scikit-learn matplotlib pillow
```

### 2. CUB 数据准备

从 CUB 数据集中选取若干类别（本项目为 20 类子集），按以下目录结构摆放：

```
CUB_200_subset/
├── images/
│   ├── 001.Black_footed_Albatross/
│   │   ├── Black_Footed_Albatross_0001_796148.jpg
│   │   └── ...
│   └── 002.Laysan_Albatross/
│       └── ...
└── text/
    ├── 001.Black_footed_Albatross/
    │   ├── Black_Footed_Albatross_0001_796148.txt
    │   └── ...
    └── 002.Laysan_Albatross/
        └── ...
```

每个 `.txt` 文件包含 10 句鸟类描述（每行一句），与图片同名。

## 🚀 使用流程

### Step 1：数据预处理

```bash
python convert_cub_json.py
```

扫描 `images/` 和 `text/` 目录，自动生成 `annotations_train/val/test.json`。

### Step 2：训练单模态（文本基线）

```bash
python singleModal_TextOnly.py
```

### Step 3：训练多模态 Late Fusion

```bash
python multimodal_latefusion.py
```

训练过程中会自动保存：
- 最优模型权重 → `text_best.pth` / `multimodal_best.pth`
- 训练曲线 → `text_only_curve.png` / `multimodal_curve.png`

## ⚙️ 训练参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| 优化器 | AdamW | — |
| 学习率 | 1e-4 | — |
| Batch Size | 4 | — |
| Epoch | 10 | — |
| 图像尺寸 | 224 × 224 | ResNet 输入要求 |
| 文本 max_len | 32 | BERT 截断长度 |
| Dropout | 0.2 | 融合分类器防止过拟合 |
| 损失函数 | CrossEntropyLoss | — |

## 📊 评估指标

- **Accuracy**：整体分类准确率
- **Macro-F1**：各类别 F1 分数的算术平均，衡量类别均衡下的分类性能

## 🤝 数据增强策略

| 模态 | 策略 | 实现位置 |
|---|---|---|
| 图像 | Resize + Normalize | `multimodal_latefusion.py:20-29` |
| 文本 | 每条样本随机选 1/10 caption | `multimodal_latefusion.py:51-52` |

## 💡 注意事项

1. **国内网络环境**：代码已内置 HuggingFace 镜像 `hf-mirror.com`，无需额外配置即可下载 BERT 权重
2. **路径硬编码**：当前代码中数据集路径写死在 `D:\kc_bird\...`，迁移到其他机器时请修改 `multimodal_latefusion.py:18`、`:90-92` 和 `singleModal_TextOnly.py:61-63`
3. **GPU 自动检测**：`device = "cuda" if torch.cuda.is_available() else "cpu"`，无 GPU 时自动使用 CPU

## 📝 License

MIT