# CIFAR10_Classification

本项目基于 PyTorch 框架，使用卷积神经网络（CNN）与 ResNet18 结构对 CIFAR-10 图像分类任务进行训练、评估与可视化分析，支持多种损失函数与模型结构的灵活配置，并提供丰富的结果可视化功能（如 t-SNE、Grad-CAM、混淆矩阵等）。

---

## 项目结构

```
.
├── data/
│   └── dataloader.py           # CIFAR-10 数据加载器封装
├── model/
│   ├── cnn.py                  # 自定义 CNN 网络结构
│   └── resnet.py               # 轻量化 ResNet18 网络结构
├── utils/
│   ├── utils.py                # 可视化工具函数、Grad-CAM、t-SNE、保存日志等
│   └── loss.py                 # Focal Loss 与 Label Smoothing 实现
├── train.py                    # 训练主脚本
├── evaluate.py                 # 评估主脚本（生成混淆矩阵、t-SNE、Grad-CAM 等）
├── config.yaml                 # 模型和训练参数配置文件（需自建）
└── README.md                   # 项目说明文档
```

---

## 安装依赖

```bash
pip install torch torchvision matplotlib scikit-learn seaborn opencv-python pyyaml
```

---

## 支持的模型结构

- `cnn`: 三层卷积 + BN + 激活（ReLU/ELU/LeakyReLU/GELU）+ FC + Dropout【29†source】
- `resnet18`: 自定义轻量化 ResNet18 实现，适配 32x32 输入【30†source】

---

## 支持的损失函数【32†source】

通过 `config.yaml` 中 `loss_function` 字段选择：

- `crossentropy`：默认标准交叉熵
- `labelsmoothing`：Label Smoothing CrossEntropy（平滑正则）
- `focal`：Focal Loss，适合处理类不平衡情况

---

## 配置文件（config.yaml）

示例：

```yaml
model:
  name: cnn
  dropout: 0.5
  activation: relu
  num_classes: 10

training:
  learning_rate: 0.001
  batch_size: 64
  epochs: 20
  optimizer: AdamW
  loss_function: CrossEntropy

data:
  num_workers: 2
```

---

## 训练模型【28†source】

```bash
python train.py
```

训练过程中日志与模型将保存在 `runs/时间戳/` 目录下，包括：

- `checkpoints/`: 每轮保存模型
- `logs/`: 文本与 CSV 格式训练日志
- `outputs/loss_curve.png`: 损失曲线图

---

## 模型评估与可视化【27†source】

```bash
python evaluate.py --checkpoint runs/时间戳/checkpoints/best_model.pt
```

自动生成：

- 分类准确率、精度报告：`classification_report.txt`
- 混淆矩阵图：`confusion_matrix.png`
- 各类精度柱状图：`class_accuracy.png`
- 错误分类示例：`misclassified.png`
- 特征 t-SNE 降维可视化：`feature_tsne.png`
- Grad-CAM 热力图（错误样本）：`gradcam/*.png`

---

## 可视化样例

| 图像名                     | 含义                        |
|----------------------------|-----------------------------|
| `loss_curve.png`           | 训练损失随 epoch 变化曲线   |
| `confusion_matrix.png`     | 测试集混淆矩阵               |
| `class_accuracy.png`       | 每一类分类精度柱状图         |
| `misclassified.png`        | 预测错误的样本示例           |
| `feature_tsne.png`         | 流形特征可视化（t-SNE）      |
| `gradcam/*.png`            | Grad-CAM 热力图              |

---

## 示例命令（完整流程）

```bash
python train.py
python evaluate.py --checkpoint runs/20250610_103000/checkpoints/best_model.pt
```

---

## 联系方式

欢迎提交 issue 或联系作者进行改进建议或项目交流。

---
