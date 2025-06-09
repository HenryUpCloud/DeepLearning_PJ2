import os
import time
import csv
import yaml
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim

from model.cnn import CNN
from model.resnet import ResNet18
from data.dataloader import get_dataloaders
from utils.utils import save_checkpoint, plot_loss_curve
from utils.loss import LabelSmoothingCrossEntropy, FocalLoss

def get_model(name, config):
    name = name.lower()
    if name == "cnn":
        return CNN(
            dropout=config['model']['dropout'],
            activation_name=config['model'].get('activation', 'relu')  # 默认使用 ReLU
        )
    elif name == "resnet18":
        return ResNet18(num_classes=config['model']['num_classes'])
    else:
        raise ValueError(f"Unknown model name: {name}")
    
def get_scheduler(optimizer, config):
    if config['training']['optimizer'] == "SGD":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['training'].get('lr_step_size', 5),
            gamma=config['training'].get('lr_gamma', 0.5)
        )
    else:
        return None
    
def get_loss_function(name):
    name = name.lower()
    if name == 'crossentropy':
        return nn.CrossEntropyLoss()
    elif name == 'labelsmoothing':
        return LabelSmoothingCrossEntropy(smoothing=0.1)
    elif name == 'focal':
        return FocalLoss(gamma=2.0)
    else:
        raise ValueError(f"Unsupported loss function: {name}")

def train_model(config, run_dir):
    # 构建目录
    log_dir = os.path.join(run_dir, "logs")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    output_dir = os.path.join(run_dir, "outputs")
    for d in [log_dir, ckpt_dir, output_dir]:
        os.makedirs(d, exist_ok=True)

    # 训练参数
    lr = config['training']['learning_rate']
    epochs = config['training']['epochs']
    batch_size = config['training']['batch_size']
    dropout = config['model']['dropout']
    optimizer_name = config['training']['optimizer']
    num_workers = config['data']['num_workers']

    # 设备
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Using device: {device}")

    # 数据加载器
    trainloader, _ = get_dataloaders(batch_size=batch_size, num_workers=num_workers)

    # 模型与优化器
    model = get_model(config['model']['name'], config).to(device)
    model_name = config['model']['name'].lower()
    loss_name = config['training'].get('loss_function', 'crossentropy')
    criterion = get_loss_function(loss_name)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    scheduler = get_scheduler(optimizer, config)

    # 日志路径
    txt_log_path = os.path.join(log_dir, "train_log.txt")
    csv_log_path = os.path.join(log_dir, "train_log.csv")

    # 日志与训练主循环
    losses = []
    best_loss = float('inf')

    with open(txt_log_path, "w") as txt_file, open(csv_log_path, "w", newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Epoch", "Loss", "EpochTime(s)"])
        total_start_time = time.time()

        for epoch in range(epochs):
            epoch_start_time = time.time()
            total_loss = 0

            model.train()
            for inputs, labels in trainloader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # 学习率调度步进（如果有 scheduler）
            if scheduler is not None:
                scheduler.step()

            avg_loss = total_loss / len(trainloader)
            losses.append(avg_loss)
            epoch_time = time.time() - epoch_start_time

            print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s")
            txt_file.write(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s\n")
            csv_writer.writerow([epoch+1, avg_loss, round(epoch_time, 2)])

            # 保存当前轮模型
            ckpt_path = os.path.join(ckpt_dir, f"{model_name}_epoch{epoch+1}.pt")
            save_checkpoint(model, optimizer, epoch, ckpt_path)
            
            # 保存最佳模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_ckpt_path = os.path.join(ckpt_dir, "best_model.pt")
                save_checkpoint(model, optimizer, epoch, best_ckpt_path)

        total_time = time.time() - total_start_time
        print(f"\nTotal Training Time: {total_time:.2f} seconds")
        plot_loss_curve(losses, os.path.join(output_dir, "loss_curve.png"))

if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join("runs", timestamp)
    os.makedirs(run_dir, exist_ok=True)

    train_model(config, run_dir)
