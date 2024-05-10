import os
from astropy.io import fits
from PIL import Image
import numpy as np
from torchvision import transforms
import torchvision.transforms.functional as F
from torch.utils.data import Dataset, DataLoader
import torch
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer, TrainingArguments, Trainer, ViTImageProcessor

class CustomFitsDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.images = [os.path.join(img_dir, file) for file in os.listdir(img_dir) if file.endswith('.fits')]
        self.labels = [os.path.join(label_dir, file) for file in os.listdir(label_dir) if file.endswith('.txt')]
        
        # Debugging output
        print("Found FITS files:", self.images)
        print("Found label files:", self.labels)

        # Ensure aligned order of images and labels if they are named correspondingly
        self.images.sort()
        self.labels.sort()

    def __len__(self):
        return len(self.images)

    def manual_pad(self,image_tensor, padding):
    # Assuming padding = (left, right, top, bottom)
        _, h, w = image_tensor.shape
        new_h = h + padding[2] + padding[3]
        new_w = w + padding[0] + padding[1]
        new_image = torch.zeros((3, new_h, new_w), dtype=image_tensor.dtype)
        new_image[:, padding[2]:padding[2]+h, padding[0]:padding[0]+w] = image_tensor
        return new_image

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label_path = self.labels[idx]
        label = None
        with fits.open(img_path) as hdul:
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


        with open(label_path, 'r') as f:
            for line in f:
                if "Object Type" in line:
                    label = line.split('= ')[1].split('\n')[0]
                    #print(label)
                    break
        if label is None:
            return None
        
        return image_tensor, label

# # Define a function to handle tensor transformations, if necessary
# def tensor_transform(image):
#     try:
#         # Ensure the image tensor has the correct shape (C, H, W)
#         if image.ndim == 4 and image.shape[0] == 1:  # (B, C, H, W) where B is 1
#             image = image.squeeze(0)  # Remove batch dimension if batch size is 1
#         c, h, w = image.shape
#         if h < 224 or w < 224:
#             # Calculate padding
#             pad_h = (224 - h) if h < 224 else 0
#             pad_w = (224 - w) if w < 224 else 0
#             padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
#             image = F.pad(image, padding, fill=0, padding_mode='constant')
#         # Resize the image (only if necessary)
#         image = F.resize(image, [224, 224])
#     except ValueError:
#         # Handle cases where the image does not meet the required dimensions
#         return None  # Or some form of logging or handling mechanism
#     return image

# Create the dataset
dataset = CustomFitsDataset(img_dir='images', label_dir='labels')
eval_dataset = CustomFitsDataset(img_dir='test', label_dir='testlabels')


# Load the pretrained model and tokenizer
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")#"/data_ssd/not_backed_up/akislam/Astr-ml/best_model_2")#
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")#"/data_ssd/not_backed_up/akislam/Astr-ml/best_model_2")#
tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.pad_token
tokenizer.padding_side = "right"
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Set the model to the appropriate device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)



# def preprocess_fits_image(fits_path):
#     with fits.open(fits_path) as hdul:
#         image_data = hdul[0].data
#         image_data = np.nan_to_num(image_data)
#         image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))
#         image_data = np.clip(image_data, 0, 1)
#         # Stack the single channel to create 3 identical channels
#         image_data = np.stack([image_data, image_data, image_data], axis=-1)
#         # Ensure the shape is [channels, height, width]
#         image_data = np.transpose(image_data, (2, 0, 1))
#         return image_data

def collate_fn(batch):
    batch = [b for b in batch if b is not None]  # Filter out None entries
    if not batch:
        return None  # Return None if batch is empty after filtering

    images, texts = zip(*batch)
    #images = torch.stack(images, dim=0)
    pixel_values = feature_extractor(images=images, return_tensors="pt",do_rescale=False).pixel_values
    processed_inputs = [tokenizer(text, padding="max_length", max_length=20, truncation=True, return_tensors="pt") for text in texts]
    labels = {key: torch.stack([dic[key] for dic in processed_inputs]) for key in processed_inputs[0]}
    return {"pixel_values": pixel_values, "labels": labels["input_ids"], "decoder_attention_mask": labels["attention_mask"]}

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./model_save',
    evaluation_strategy='steps',
    save_strategy='steps',
    logging_dir='./logs',
    per_device_train_batch_size=32,
    num_train_epochs=500,
    weight_decay=0.01,
    save_total_limit=20,
    fp16=torch.cuda.is_available(),
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=eval_dataset,  
    tokenizer=tokenizer,
    data_collator=collate_fn
)

# Train the model
trainer.train()

model.save_pretrained('./best_model')
tokenizer.save_pretrained('./best_model')