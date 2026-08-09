import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from dataset import get_dataloaders
from model import CNNModel

def evaluate_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Using device: {device}")

    # 1. Load Test DataLoader
    _, test_loader, classes = get_dataloaders()

    # 2. Create Model structure and load trained weights
    model = CNNModel().to(device)
    model_path = './models/best_cnn_model.pth'
    
    if not os.path.exists(model_path):
        print("❌ Saved model weights not found! Please run train.py first.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds = []
    all_targets = []

    print("📊 Generating predictions on the Test Dataset...")

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    # 3. Precision, Recall, F1-Score Report
    print("\n================ Classification Report ================\n")
    print(classification_report(all_targets, all_preds, target_names=classes))

    # 4. Generate Confusion Matrix Plot
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('CIFAR-10 CNN Confusion Matrix')

    # Save the plot
    os.makedirs('./outputs/plots', exist_ok=True)
    plot_path = './outputs/plots/confusion_matrix.png'
    plt.savefig(plot_path)
    print(f"\n📈 Confusion Matrix plot saved at: {plot_path}")

if __name__ == '__main__':
    evaluate_model()