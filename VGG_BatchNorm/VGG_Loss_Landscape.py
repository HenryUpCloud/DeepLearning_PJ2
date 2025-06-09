import matplotlib as mpl
mpl.use('Agg')  # 非GUI服务器
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import random
from tqdm import tqdm
from models.vgg import VGG_A, VGG_A_BatchNorm  # 你需要已实现 VGG_A_BatchNorm
from data.loaders import get_cifar_loader
from torch import nn

# 设置设备为 MPS（更新了MPS检查方法）
device = torch.device("mps" if torch.backends.mps.is_built() else "cpu")  # 检查是否支持 MPS 后端
figures_path = './reports/figures'
models_path = './reports/models'
os.makedirs(figures_path, exist_ok=True)
os.makedirs(models_path, exist_ok=True)

# 随机种子
def set_random_seeds(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 数据加载
train_loader = get_cifar_loader(train=True)
val_loader = get_cifar_loader(train=False)

# 训练函数
def train(model, optimizer, criterion, train_loader, epochs_n=20, model_save_path=None):
    model = model.to(device)  # 确保模型加载到正确的设备上
    model.train()
    losses = []

    for epoch in tqdm(range(epochs_n), desc="Training"):
        epoch_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)  # 将数据移到 MPS 设备上
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        losses.append(epoch_loss / len(train_loader))

        # 保存模型的权重
        if model_save_path:
            torch.save(model.state_dict(), f"{model_save_path}_epoch_{epoch+1}.pth")
            print(f"Model saved at epoch {epoch+1} to {model_save_path}_epoch_{epoch+1}.pth")
    
    return losses

# 可视化函数
def plot_loss_landscape(loss_dict, title='Loss Landscape'):
    epochs = len(next(iter(loss_dict.values())))
    x = list(range(epochs))

    max_curve = [max(losses[i] for losses in loss_dict.values()) for i in x]
    min_curve = [min(losses[i] for losses in loss_dict.values()) for i in x]

    plt.figure(figsize=(10, 5))
    for lr, losses in loss_dict.items():
        plt.plot(x, losses, label=f'lr={lr}')
    plt.fill_between(x, min_curve, max_curve, alpha=0.3, label='Variance region')
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(figures_path, f'{title.replace(" ", "_")}.png'))

# 主流程
def run_experiment(model_class, name):
    learning_rates = [1e-3, 2e-3, 1e-4, 5e-4]
    results = {}

    for lr in learning_rates:
        print(f'\n{name} with lr={lr}')
        set_random_seeds(42)
        model = model_class()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        # 指定模型保存路径
        model_save_path = os.path.join(models_path, f"{name}_lr_{lr:.0e}")

        # 训练并保存模型
        loss = train(model, optimizer, criterion, train_loader, model_save_path=model_save_path)
        results[lr] = loss

    plot_loss_landscape(results, title=f'Loss Landscape ({name})')

# 运行两个模型
if __name__ == "__main__":
    run_experiment(VGG_A, "VGG-A")
    run_experiment(VGG_A_BatchNorm, "VGG-A with BN")
