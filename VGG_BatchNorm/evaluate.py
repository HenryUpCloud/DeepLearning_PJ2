import torch
import os
from models.vgg import VGG_A, VGG_A_BatchNorm  # 导入你已实现的 VGG 模型
from data.loaders import get_cifar_loader
from torch import nn
import argparse

# 路径配置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 确保使用 CUDA 设备
models_path = './reports/models'
figures_path = './reports/figures'

# 加载测试集
val_loader = get_cifar_loader(train=False, n_items=100)

# 测试函数
def evaluate(model, test_loader):
    model.eval()  # 设置为评估模式
    correct = 0
    total = 0
    test_loss = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():  # 不需要计算梯度
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            test_loss += loss.item()

            _, predicted = torch.max(out, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    accuracy = 100 * correct / total
    avg_loss = test_loss / len(test_loader)
    return accuracy, avg_loss

# 加载模型并评估
def load_and_evaluate(model_class, model_path):
    model = model_class().to(device)  # 初始化模型并移到设备
    model.load_state_dict(torch.load(model_path))  # 加载保存的模型权重
    print(f"Evaluating model: {model_path}")
    
    accuracy, avg_loss = evaluate(model, val_loader)
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Test Loss: {avg_loss:.4f}")

# 主函数
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained model on the test set")
    parser.add_argument('--model', type=str, required=True, help="The model to evaluate (VGG_A or VGG_A_BatchNorm)")
    parser.add_argument('--rate', type=str, required=True, help="The learning rate")
    parser.add_argument('--epoch', type=int, required=True, help="Epoch number of the model to evaluate (e.g., 10)")
    args = parser.parse_args()

    # 模型选择
    if args.model == 'VGG-A':
        model_class = VGG_A
    elif args.model == 'VGG-A with BN':
        model_class = VGG_A_BatchNorm
    else:
        raise ValueError("Invalid model name. Choose 'VGG_A' or 'VGG_A_BatchNorm'.")

    # 模型路径
    model_path = os.path.join(models_path, f"{args.model}_lr_{args.rate}_epoch_{args.epoch}.pth")  # 使用相应学习率和 epoch

    # 评估模型
    if os.path.exists(model_path):
        load_and_evaluate(model_class, model_path)
    else:
        print(f"Model file {model_path} does not exist. Please check the path or epoch number.")
