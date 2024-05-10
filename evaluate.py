import os
from astropy.io import fits
import numpy as np
import torch
from transformers import AutoTokenizer, VisionEncoderDecoderModel, ViTFeatureExtractor, ViTImageProcessor, AutoModel
import torchvision.transforms.functional as F
# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model and tokenizer
# model = VisionEncoderDecoderModel.from_pretrained('nlpconnect/vit-gpt2-image-captioning').to(device)
# tokenizer = AutoTokenizer.from_pretrained('nlpconnect/vit-gpt2-image-captioning')

model = VisionEncoderDecoderModel.from_pretrained('/data_ssd/not_backed_up/akislam/Astr-ml/model_save_past2/checkpoint-9500').to(device)
tokenizer = AutoTokenizer.from_pretrained('/data_ssd/not_backed_up/akislam/Astr-ml/model_save_past2/checkpoint-9500')

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
def manual_pad(image_tensor, padding):
    # Assuming padding = (left, right, top, bottom)
        _, h, w = image_tensor.shape
        new_h = h + padding[2] + padding[3]
        new_w = w + padding[0] + padding[1]
        new_image = torch.zeros((3, new_h, new_w), dtype=image_tensor.dtype)
        new_image[:, padding[2]:padding[2]+h, padding[0]:padding[0]+w] = image_tensor
        return new_image
# Function to process and prepare the image
def prepare_image(image_path):
    with fits.open(image_path) as hdul:
        image_data = hdul[0].data.astype(np.float32)

        # Check if the image is 2D or 3D with 1 or 3 channels
        if image_data.ndim == 2:  # 2D image
            image_data = np.stack([image_data] * 3, axis=0)  # Convert to 3-channel
        elif image_data.ndim == 3 and (image_data.shape[0] == 1 or image_data.shape[0] == 3):
            if image_data.shape[0] == 1:  # Single channel
                image_data = np.repeat(image_data, 3, axis=0)  # Repeat channel to make it 3-channel
            elif image_data.shape[0] == 3:
                # If it's already 3 channels, we assume it's in the correct format
                pass
        else:
            return None  # Skip images that do not meet the criteria

        # Normalize the image data
        image_data = np.nan_to_num(image_data)
        min_val = np.min(image_data)
        max_val = np.max(image_data)
        if max_val > min_val:  # Avoid division by zero
            image_data = (image_data - min_val) / (max_val - min_val)
        else:
            return None  # Skip this image if no variation in pixel values

        image_tensor = torch.from_numpy(image_data).float()  # Convert to tensor
    return image_tensor
# Evaluate the model
def evaluate_model(test_dir, label_dir):
    for filename in os.listdir(test_dir):
        if filename.endswith('.fits'):
            image_path = os.path.join(test_dir, filename)
            label_path = os.path.join(label_dir, filename.replace('image.fits', 'label.txt'))

            # Prepare image
            image_tensor = prepare_image(image_path).to(device)
            pixel_values = feature_extractor(images=image_tensor, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(device)

            # Generate prediction
            outputs = model.generate(pixel_values, max_new_tokens=20)
            pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Read actual label
            with open(label_path, 'r') as file:
                for line in file:
                    if "Object Type" in line:
                        actual_label = line.split('= ')[1].strip()
                        break

            print(f"Filename: {filename}")
            print(f"Predicted: {pred_text}")
            print(f"Actual: {actual_label}\n")

# Paths to the test directory and label directory
test_dir = 'images/'
label_dir = 'labels/'

# Call the evaluation function
evaluate_model(test_dir, label_dir)
