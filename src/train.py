import os
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders
from model import CNNModel

def train_model(epochs=5, batch_size=64, learning_rate=0.001):
    # Device check (Will run faster if GPU is available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Using device: {device}")

    # 1. Load the data
    train_loader, test_loader, classes = get_dataloaders(batch_size=batch_size)

    # 2. Setup Model, Loss Function, and Optimizer
    model = CNNModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_acc = 0.0
    os.makedirs('./models', exist_ok=True)

    print("\n🚀 Training is starting...\n")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass & Optimize (Learn and update weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100. * correct / total
        avg_loss = running_loss / len(train_loader)

        # Testing/Validation Step (Check how well the model learned)
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()

        test_acc = 100. * test_correct / test_total

        print(f"Epoch [{epoch+1}/{epochs}] -> Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

        # Save model weights if test accuracy improves
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), './models/best_cnn_model.pth')
            print(f"   💾 Model weights saved into 'models/best_cnn_model.pth'! (Best Test Acc: {best_acc:.2f}%)")

    print("\n🎉 Training Complete!")

if __name__ == '__main__':
    # Train for 5 epochs
    train_model(epochs=5)