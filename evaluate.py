"""
evaluate.py
Loads the best saved model and generates a confusion matrix + classification
report on the validation set.

Run from VS Code terminal:
    python evaluate.py
"""

import torch
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

from dataset import get_dataloaders
from model import build_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    _, val_loader, class_names = get_dataloaders()

    model = build_model(num_classes=len(class_names))
    model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Car Angle Classification - Confusion Matrix")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("Saved confusion_matrix.png")


if __name__ == "__main__":
    main()