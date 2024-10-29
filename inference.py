from model import MobileNetInspiredDetector  # Import your model
import torch
from trainfxn import *  # Import utility functions
import numpy as np
from PIL import Image
import pandas as pd  # This is just an example if pandas usage is needed
test_path = "test_path"
# Load model and weights
weights_path = "v2weights.ckpt"  # Path to weights file
weights = torch.load(weights_path, weights_only=True)
model = MobileNetInspiredDetector(num_classes=1, boxes=5, input_ch=3)
model.load_state_dict(weights['model_state_dict'])

# Load and process a random image
image_files = get_image_files(test_path)  # Directory containing test images
selected_image_name = pick_random_image(image_files)
selected_image_path = f"{test_path}/{selected_image_name}"
img = Image.open(selected_image_path).convert('RGB')  # Open and convert to RGB

# Convert image to tensor format compatible with model
img_array, img_tensor, display_tensor = img_to_tensor(img)
display_image = img_tensor.permute(1, 2, 0)  # Permute tensor dimensions for display

# Run inference and get predicted bounding boxes and scores
output = model.forward(display_image)
predicted_boxes = output[0].squeeze(0)
predicted_scores = output[1].squeeze(0).squeeze(1)

# Sort top k bounding boxes and scores for display
top_k = 5
top_boxes, top_scores, labels = topk_sort(top_k, predicted_boxes, predicted_scores)
labels[1] += "img1"  # Add label for first image (or customize as needed)

# Annotate image with bounding boxes and labels, then display
annotated_image = tensor_to_image_with_boxes(display_tensor, labels, top_boxes)
annotated_image.show()
