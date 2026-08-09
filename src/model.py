import torch
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        
        # 1. Feature Extraction Layers 
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Image size: 32x32 -> 16x16

            # Block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Image size: 16x16 -> 8x8
        )
        
        # 2. Classification Layers 
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.4), # Prevents the model from memorizing (overfitting)
            nn.Linear(256, 10) # Scores for 10 classes
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    # Model Test Run
    model = CNNModel()
    dummy_input = torch.randn(2, 3, 32, 32) # 2 dummy images for testing
    output = model(dummy_input)
    
    print(" CNN Model structure is completely correct!")
    print(f"Output Shape (Batch Size, Classes): {output.shape}")