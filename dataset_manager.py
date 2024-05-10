import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from astropy.io import fits
from collections import Counter
from transformers import Trainer

class CustomFitsDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None, min_samples_per_class=10):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.min_samples = min_samples_per_class

        # Load image and label paths
        self.images = [os.path.join(img_dir, file) for file in os.listdir(img_dir) if file.endswith('.fits')]
        self.labels = [os.path.join(label_dir, file) for file in os.listdir(label_dir) if file.endswith('.txt')]

        # Create label map and filter labels
        self.label_map, self.reverse_label_map, self.num_classes = self.create_label_map()
        self.filter_labels_and_images()

        # Compute class weights for handling class imbalance
        label_indices = [self.label_map[self.extract_label(lp)] for lp in self.labels if self.extract_label(lp) in self.label_map]
        self.label_count = Counter(label_indices)
        total_count = sum(self.label_count.values())
        self.weights = [total_count / self.label_count.get(idx, 1) for idx in label_indices]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label_path = self.labels[idx]
        label = self.extract_label(label_path)
        if label is None or label not in self.label_map:
            return None

        with fits.open(img_path) as hdul:
            image_data = hdul[0].data.astype(np.float32)
            if image_data.ndim != 2 and (image_data.ndim != 3 or image_data.shape[0] not in {1, 3}):
                return None

            if image_data.ndim == 3 and image_data.shape[0] == 1:
                image_data = np.repeat(image_data, 3, axis=0)
            elif image_data.ndim == 2:
                image_data = np.stack([image_data] * 3, axis=0)

            image_data = np.nan_to_num(image_data)
            min_val, max_val = np.min(image_data), np.max(image_data)
            image_data = (image_data - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(image_data)

            image_tensor = torch.from_numpy(image_data).float()
            if self.transform:
                image_tensor = self.transform(image_tensor)

        label_idx = self.label_map[label]
        return image_tensor, label_idx

    def extract_label(self, label_path):
        try:
            with open(label_path, 'r') as file:
                for line in file:
                    if "Object Type" in line:
                        return line.split('= ')[1].strip()
        except FileNotFoundError:
            return None

    def create_label_map(self):
        temp_label_count = Counter()
        for label_path in self.labels:
            label = self.extract_label(label_path)
            if label:
                temp_label_count[label] += 1

        filtered_labels = {label for label, count in temp_label_count.items() if count >= self.min_samples}
        label_map = {label: idx for idx, label in enumerate(sorted(filtered_labels))}
        reverse_map = {idx: label for label, idx in label_map.items()}
        num_classes = len(label_map)
        return label_map, reverse_map, num_classes

    def filter_labels_and_images(self):
        filtered_images = []
        filtered_labels = []
        for img_path, label_path in zip(self.images, self.labels):
            label = self.extract_label(label_path)
            if label in self.label_map:
                filtered_images.append(img_path)
                filtered_labels.append(label_path)
        self.images = filtered_images
        self.labels = filtered_labels


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        weights = inputs.pop("weights")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=inputs.get('weights', None))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

