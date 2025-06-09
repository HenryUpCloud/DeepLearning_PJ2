import os
import re
import torch
import yaml
import argparse
import numpy as np
from glob import glob
from sklearn.metrics import classification_report

from model.cnn import CNN
from model.resnet import ResNet18
from data.dataloader import get_dataloaders
from utils.utils import (
    plot_confusion_matrix,
    plot_class_accuracy,
    show_misclassified_images,
    plot_tsne,
    generate_gradcam,
    save_gradcam_image
)

def get_model(config):
    model_name = config['model']['name'].lower()
    if model_name == "cnn":
        return CNN(dropout=config['model']['dropout'])
    elif model_name == "resnet18":
        return ResNet18(num_classes=config['model']['num_classes'])
    else:
        raise ValueError(f"Unknown model: {model_name}")

def get_target_layer(model, config):
    name = config['model']['name'].lower()
    if name == 'cnn':
        return model.conv3  # 你原来的 CNN 模型中最后一个卷积层
    elif name == 'resnet18':
        return model.layer4[-1].conv2  # ResNet 最后一块的第二个卷积
    else:
        raise ValueError(f"Grad-CAM target layer undefined for model: {name}")

def evaluate(config, checkpoint_path):
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Using device: {device}")

    _, testloader = get_dataloaders(
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers']
    )

    model = get_model(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    all_preds, all_labels, all_imgs = [], [], []

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_imgs.extend(inputs.cpu().numpy())

    class_names = testloader.dataset.classes
    acc = 100 * np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"Test Accuracy: {acc:.2f}%")

    output_dir = os.path.dirname(checkpoint_path).replace("checkpoints", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # 保存分类报告
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(report)
    print("classification_report.txt saved.")

    # 可视化
    plot_confusion_matrix(all_labels, all_preds, class_names, os.path.join(output_dir, "confusion_matrix.png"))
    print("confusion_matrix.png saved.")
    plot_class_accuracy(all_labels, all_preds, class_names, os.path.join(output_dir, "class_accuracy.png"))
    print("class_accuracy.png saved.")
    show_misclassified_images(all_imgs, all_labels, all_preds, class_names, os.path.join(output_dir, "misclassified.png"))
    print("misclassified.png saved.")
    plot_tsne(model, testloader, device, class_names, os.path.join(output_dir, "feature_tsne.png"))
    print("feature_tsne.png saved.")

    # Grad-CAM 可视化
    print("Generating Grad-CAM visualizations...")
    gradcam_dir = os.path.join(output_dir, "gradcam")
    os.makedirs(gradcam_dir, exist_ok=True)

    target_layer = get_target_layer(model, config)
    shown = 0
    for img, label, pred in zip(all_imgs, all_labels, all_preds):
        if label != pred and shown < 10:
            # 确保 img 是 [3, 32, 32] 的 numpy 数组
            if isinstance(img, np.ndarray):
                # 转成 tensor 并加 batch 维度 → [1, 3, 32, 32]
                img_tensor = torch.from_numpy(img).float().unsqueeze(0).to(device)
            elif isinstance(img, torch.Tensor):
                # 如果已经是 Tensor，确保是 [3, 32, 32]
                img_tensor = img.float().unsqueeze(0).to(device)
            else:
                raise TypeError("Unsupported image type for Grad-CAM")

            # 生成 Grad-CAM
            cam = generate_gradcam(model, img_tensor, pred, target_layer, device)

            # 保存 Grad-CAM 图像（注意 squeeze 回 [3, 32, 32]）
            save_path = os.path.join(
                gradcam_dir,
                f"wrong_{shown}_T{class_names[label]}_P{class_names[pred]}.png"
            )
            save_gradcam_image(img_tensor.squeeze(0).cpu(), cam, save_path)
            shown += 1


    '''
    shown = 0
    for img, label, pred in zip(all_imgs, all_labels, all_preds):
        if label != pred and shown < 10:
            img_tensor = torch.tensor(img).unsqueeze(0).to(device)
            cam = generate_gradcam(model, img_tensor, pred, target_layer, device)
            save_path = os.path.join(gradcam_dir, f"wrong_{shown}_T{class_names[label]}_P{class_names[pred]}.png")
            save_gradcam_image(img_tensor.squeeze(0).cpu(), cam, save_path)
            shown += 1
    print(f"Saved {shown} Grad-CAM images to {gradcam_dir}")
    '''

def find_latest_checkpoint(checkpoint_dir="runs"):
    run_dirs = sorted(glob(os.path.join(checkpoint_dir, "*")), reverse=True)
    epoch_pattern = re.compile(r"epoch(\d+)\.pt")

    for run_dir in run_dirs:
        ckpt_path = os.path.join(run_dir, "checkpoints")
        ckpt_files = glob(os.path.join(ckpt_path, "*.pt"))
        
        if ckpt_files:
            # 从文件名中提取 epoch 数字，并按数值降序排列
            ckpt_files.sort(key=lambda x: int(epoch_pattern.search(x).group(1)) if epoch_pattern.search(x) else -1, reverse=True)
            return ckpt_files[0]

    raise FileNotFoundError("No checkpoint found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    args = parser.parse_args()

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    ckpt = args.checkpoint or find_latest_checkpoint()
    print(f"Loading checkpoint: {ckpt}")
    evaluate(config, ckpt)
