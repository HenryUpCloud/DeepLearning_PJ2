import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.manifold import TSNE
import seaborn as sns

def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)

def plot_loss_curve(losses, path):
    plt.figure()
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.savefig(path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names, path):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_class_accuracy(y_true, y_pred, class_names, path):
    accs = []
    for i in range(len(class_names)):
        idx = np.array(y_true) == i
        correct = np.sum((np.array(y_pred)[idx] == i))
        total = np.sum(idx)
        accs.append(correct / total if total else 0)

    plt.figure(figsize=(10, 4))
    plt.bar(class_names, accs, color='skyblue')
    plt.ylabel('Accuracy')
    plt.title('Per-Class Accuracy')
    plt.ylim(0, 1.0)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def show_misclassified_images(images, labels, preds, class_names, save_path, max_per_class=5):
    wrong_idx = (np.array(labels) != np.array(preds))
    mis_imgs = np.array(images)[wrong_idx]
    mis_lbls = np.array(labels)[wrong_idx]
    mis_preds = np.array(preds)[wrong_idx]

    shown = {cls: 0 for cls in range(len(class_names))}
    fig, axes = plt.subplots(len(class_names), max_per_class, figsize=(max_per_class * 2, len(class_names) * 2))

    for img, lbl, pred in zip(mis_imgs, mis_lbls, mis_preds):
        if shown[lbl] >= max_per_class:
            continue
        ax = axes[lbl, shown[lbl]]
        img = img.transpose(1, 2, 0) * 0.5 + 0.5  # unnormalize
        ax.imshow(img)
        ax.set_title(f'T:{class_names[lbl]}\nP:{class_names[pred]}', fontsize=8)
        ax.axis('off')
        shown[lbl] += 1

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_tsne(model, dataloader, device, class_names, save_path):
    features, labels = [], []
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            feats = model.extract_features(x).cpu()
            features.append(feats)
            labels.extend(y.numpy())

    features = torch.cat(features, dim=0).numpy()
    tsne = TSNE(n_components=2, random_state=42).fit_transform(features)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=tsne[:, 0], y=tsne[:, 1], hue=labels, palette='tab10', s=15, legend='full')
    plt.title("t-SNE of Feature Representations")
    plt.legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Plot Grad-CAM
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt

def generate_gradcam(model, image_tensor, class_idx, target_layer, device):
    model.eval()
    image_tensor = image_tensor.to(device).requires_grad_()  # 🚫 不再重复 unsqueeze

    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output.detach())

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_full_backward_hook(backward_hook)

    output = model(image_tensor)  # input: [1, 3, 32, 32]
    model.zero_grad()
    output[0, class_idx].backward()

    handle_f.remove()
    handle_b.remove()

    act = activations[0].squeeze(0)  # shape: [C, H, W]
    grad = gradients[0].squeeze(0)   # shape: [C, H, W]
    weights = grad.mean(dim=(1, 2))  # shape: [C]
    cam = torch.sum(weights[:, None, None] * act, dim=0)  # shape: [H, W]
    cam = F.relu(cam)

    cam = cam.cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
    cam = cv2.resize(cam, (32, 32))
    return cam

def save_gradcam_image(image_tensor, cam, save_path):
    # 转成 HWC 顺序 + 反归一化 + detach
    image = image_tensor.detach().permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5
    image = (image * 255).astype("uint8")

    heatmap = (cam * 255).astype("uint8")
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    superimposed = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
    plt.imshow(superimposed)
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

