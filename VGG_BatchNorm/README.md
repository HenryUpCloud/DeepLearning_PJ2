# VGG_BatchNorm on CIFAR-10

本项目基于 PyTorch 实现 `VGG_A` 与 `VGG_A_BatchNorm` 网络结构，并在 CIFAR-10 数据集上比较二者在不同学习率下的损失变化趋势（Loss Landscape），以研究 Batch Normalization 对收敛速度与稳定性的影响。

---

## 项目结构

```
VGG_BatchNorm/
├── models/ 
│   └── vgg.py              # 含 VGG_A、VGG_A_BatchNorm、VGG_A_Light、VGG_A_Dropout 网络
├── VGG_Loss_Landscape.py 
├── evaluate.py             # 不同学习率下的训练与 Loss 可视化主脚本
├── utils/
│   └── nn.py               # Xavier 初始化等权重工具函数
├── data/
│   └── loaders.py          # CIFAR-10 数据加载封装（需自建）
├── reports/
│   ├── figures/            # 保存 Loss 曲线图
│   └── models/             # 保存每轮模型权重
└── README.md
```

---

## 支持的模型

### `VGG_A`
- 基础版本，包含 5 个 stage 的卷积与最大池化；
- 适配输入尺寸为 32x32 的 CIFAR-10 图像。

### `VGG_A_BatchNorm`
- 在每个卷积层后添加 `BatchNorm2d` 层；
- 有助于加速收敛、抑制梯度爆炸/消失。

---

## 快速开始

### 1️安装依赖

```bash
pip install torch torchvision matplotlib tqdm numpy
```

确保你有 `data/loaders.py` 文件，它应包含类似如下内容：

```python
def get_cifar_loader(train=True, batch_size=64):
    # 返回 DataLoader
```

（如你使用的是原始 `dataloader.py` 也可以适配即可）

---

### 2️运行实验（主脚本）

```bash
python VGG_Loss_Landscape.py
```

该脚本会：

- 分别训练 `VGG_A` 和 `VGG_A_BatchNorm`；
- 学习率遍历：`[1e-3, 2e-3, 1e-4, 5e-4]`；
- 每轮训练后保存模型权重（至 `./reports/models/`）；
- 自动生成 loss 曲线图（至 `./reports/figures/`）。

---

## 可视化结果

每个模型会生成一张 Loss Landscape 曲线图，例如：

- `./reports/figures/Loss_Landscape_(VGG-A).png`
- `./reports/figures/Loss_Landscape_(VGG-A with BN).png`

图中展示：
- 每个学习率下的损失曲线；
- 所有曲线间的最大值/最小值波动区域（灰色透明区域）。

---

## 模型保存

所有中间模型保存在：

```
./reports/models/VGG-A_lr_1e-03_epoch_20.pth
./reports/models/VGG-A_with_BN_lr_2e-03_epoch_20.pth
...
```

方便后续载入测试或迁移学习。

---

## 超参数与设备设置

- 默认设备支持 MPS（Apple Silicon），否则使用 CUDA 或 CPU；
- 每轮训练损失记录在 `losses` 列表中；
- 可通过修改 `VGG_Loss_Landscape.py` 的 `run_experiment()` 更改模型或超参数设置。

---

## 联系方式

欢迎在 GitHub 提 issue 交流项目设计、结构、改进建议。
```

---


