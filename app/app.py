import sys
import os
import torch
import torchvision.transforms as transforms
import gradio as gr
from PIL import Image

# Add src directory to system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from model import CNNModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model structure and trained weights
model = CNNModel().to(device)
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/best_cnn_model.pth'))

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

def predict(img):
    if img is None:
        return None
    image = Image.fromarray(img).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    confidences = {classes[i]: float(probabilities[i]) for i in range(10)}
    return confidences

# Gradio Web Interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(),
    outputs=gr.Label(num_top_classes=3),
    title="🖼️ CNN Image Classification System",
    description="Upload an image (Car, Dog, Cat, Plane, Ship, etc.) to get real-time predictions!"
)

if __name__ == '__main__':
    demo.launch()