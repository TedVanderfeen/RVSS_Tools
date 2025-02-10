import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import os
import sys
from model import Net ##Model definition must be changed to match your actual model
import cv2

def load_model(model_path, device):
    model = Net().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def load_and_preprocess_image(image_path, target_size=(240, 320)): ##CHANGE TO MATCH YOUR INPUT SIZE
    """Preprocess image exactly as in deploy.py"""
    ##MUST ADD YOUR OWN PREPROCESSING STEPS HERE ##
    img_tensor = 0
    
    return img_tensor

def extract_ground_truth(filename):
    """Extract angle from filename in format '000033_0.20.jpg'"""
    try:
        parts = filename.split('_')
        if len(parts) == 2:  # Expect exactly 2 parts: ID and angle with extension
            angle = float(parts[1].split('.jpg')[0])  # Remove .jpg and convert to float
            return angle
    except ValueError:
        return None
    return None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Test model in PyTorch')
    parser.add_argument('--model_path', type=str, 
                       default=os.path.join(os.path.dirname(__file__), "ADAM_Models", "best_model_10.pth"), 
                       help='Path to the model weights')
    parser.add_argument('--folder_path', type=str, required=True, 
                       help='Path to folder containing test images')
    parser.add_argument('--device', type=str, default='cpu', 
                       help='Device to run the model (cpu or cuda)')
    args = parser.parse_args()

    # Ensure paths are absolute
    model_path = os.path.abspath(args.model_path)
    folder_path = os.path.abspath(args.folder_path)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Image folder not found at: {folder_path}")

    device = torch.device(args.device)

    print(f"Looking for model at: {model_path}")
    print(f"Looking for images in: {folder_path}")
    print(f"Using device: {device}")

    # Debug path resolution
    print(f"Current working directory: {os.getcwd()}")
    print(f"Absolute folder path: {folder_path}")
    
    # Check if path exists and is accessible
    if not os.path.exists(folder_path):
        print(f"Path {folder_path} does not exist")
        sys.exit(1)
    if not os.access(folder_path, os.R_OK):
        print(f"Path {folder_path} is not readable")
        sys.exit(1)
        
    try:
        files = os.listdir(folder_path)
        print(f"Directory contents: {files[:5]}...")  # Show first 5 files
    except Exception as e:
        print(f"Error accessing directory: {e}")
        sys.exit(1)

    model = load_model(model_path, device)
    print("Model loaded successfully")

    # Get list of image files first
    image_files = sorted([f for f in os.listdir(folder_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print(f"Found {len(image_files)} images")

    if not image_files:
        print("No valid images found in folder")
        sys.exit(1)

    # Process images
    total_error = 0
    count = 0
    
    for filename in image_files:
        try:
            image_path = os.path.join(folder_path, filename)
            print(f"\nProcessing: {image_path}")
            
            ground_truth = extract_ground_truth(filename)
            if ground_truth is None:
                print(f"Skipping {filename} - cannot extract ground truth")
                continue

            img_tensor = load_and_preprocess_image(image_path).to(device)
            print(f"Image tensor shape: {img_tensor.shape}")
            
            with torch.no_grad():
                output = model(img_tensor)
                predicted = output[0].cpu().numpy()

            error = np.mean(np.abs(predicted - ground_truth))
            total_error += error
            count += 1

            print(f"Predicted: [{predicted[0]:.1f}, {predicted[1]:.1f}]")
            print(f"Actual:    [{ground_truth[0]:.1f}, {ground_truth[1]:.1f}]")
            print(f"Error:     {error:.2f}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

    if count > 0:
        print(f"\nProcessed {count} images")
        print(f"Average Error: {total_error/count:.2f}")
    else:
        print("\nNo images were successfully processed")