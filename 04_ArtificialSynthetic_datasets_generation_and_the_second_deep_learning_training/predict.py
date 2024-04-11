import torch
import torchvision.transforms as transforms
from PIL import Image
from train import CustomResNet
import os


def predict_angle(image_path, model_path):
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomResNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Preprocess the image
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path)

    # Convert single-channel grayscale image to 3-channel RGB image
    image_rgb = Image.new("RGB", image.size)
    image_rgb.paste(image)

    image_tensor = transform(image_rgb).unsqueeze(0).to(device)

    # Perform prediction
    with torch.no_grad():
        output = model(image_tensor)
        predicted_angle = output.item()

    return predicted_angle


if __name__ == "__main__":
    # Specify the folder path to traverse
    folder_path = "F"
    # Define supported image file extensions
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.gif", "*.bmp"]
    model_path = "./save_weights/mos2_twist_model.pth"  # Input your model weights path
    # Traverse all files in the folder
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            predicted_angle = predict_angle(file_path, model_path)
            print(f"Predicted Angle for {file}: {predicted_angle:.2f} degrees")

