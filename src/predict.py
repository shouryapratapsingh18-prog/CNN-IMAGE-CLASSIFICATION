import os
import argparse
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from model import CNNModel

# Define CIFAR-10 class labels matching dataset.py
CLASSES = (
    'plane', 'car', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
)

# Image pre-processing matching test configuration
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])


def predict_image(image_path, model_path='./models/best_cnn_model.pth'):
    """Loads a single image, processes it through the trained CNN, and outputs the top prediction."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Verify model weights exist
    if not os.path.exists(model_path):
        print("❌ Saved model weights not found! Please run train.py first.")
        return

    # 2. Verify image path exists
    if not os.path.exists(image_path):
        print(f"❌ Image file not found: '{image_path}'")
        return

    # 3. Load and transform image
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"❌ Error opening image: {e}")
        return

    # Transform image and add batch dimension -> Shape: (1, 3, 32, 32)
    input_tensor = transform(image).unsqueeze(0).to(device)

    # 4. Load model and trained weights
    model = CNNModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 5. Run inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    predicted_label = CLASSES[predicted_idx.item()]
    confidence_score = confidence.item() * 100

    # 6. Display results
    print("\n================ Prediction Results ================")
    print(f"🖼️  Image Path: {image_path}")
    print(f"🏷️  Predicted Class: {predicted_label.upper()}")
    print(f"🎯 Confidence: {confidence_score:.2f}%")
    print("====================================================\n")

    return predicted_label, confidence_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict image class using trained CIFAR-10 CNN model.")
    parser.add_argument('--image', type=str, default='test_image.jpg', help='Path to the image file to predict')
    args = parser.parse_args()

    predict_image(image_path=args.image)