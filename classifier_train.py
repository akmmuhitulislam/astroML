from sklearn.model_selection import train_test_split
import numpy as np
import torchvision.transforms.functional as F
from dataset_manager import CustomFitsDataset, CustomTrainer
import torch
from transformers import ViTForImageClassification, TrainingArguments, ViTImageProcessor, TrainerCallback, TrainerState, TrainerControl, Trainer
from datasets import load_metric
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

dataset = CustomFitsDataset(img_dir='images/', label_dir='labels/')
labels = [dataset.extract_label(dataset.labels[i]) for i in range(len(dataset))]
label_idx = [dataset.label_map[label] for label in labels if label in dataset.label_map]

try:
    # Attempt stratified split
    train_idx, test_idx = train_test_split(range(len(label_idx)), test_size=0.2, stratify=label_idx)
except ValueError:
    # Fallback to non-stratified split if stratification fails
    print("Stratified split failed due to insufficient class samples. Using non-stratified split.")
    train_idx, test_idx = train_test_split(range(len(label_idx)), test_size=0.2)

# Creating subsets for train and test based on indices
train_dataset = torch.utils.data.Subset(dataset, train_idx)
eval_dataset = torch.utils.data.Subset(dataset, test_idx)
print(len(dataset.label_count.keys()))

# Load the feature extractor and model
feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels=dataset.num_classes, ignore_mismatched_sizes=True)

# Function to preprocess the images and prepare for the model
def preprocess_images(examples):
    pixel_values = feature_extractor(images=[x[0] for x in examples], return_tensors='pt',do_rescale=False).pixel_values
    labels = torch.tensor([x[1] for x in examples])
    return {'pixel_values': pixel_values, 'labels': labels}

class MetricsCallback(TrainerCallback):
    "A callback that logs the evaluation results."
    
    def __init__(self):
        self.train_loss_set = []
        self.eval_loss_set = []
        self.eval_accuracy_set = []
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if 'loss' in logs:
            self.train_loss_set.append(logs['loss'])
        if 'eval_loss' in logs:
            self.eval_loss_set.append(logs['eval_loss'])
        if 'eval_accuracy' in logs:
            self.eval_accuracy_set.append(logs['eval_accuracy'])

# Initialize callback
metrics_callback = MetricsCallback()

# Data collator
def collate_fn(batch):
    batch = [b for b in batch if b is not None]  # Filter out None entries
    if not batch:
        return None  # Return None if batch is empty

    images, labels = zip(*batch)
    pixel_values = feature_extractor(images=[x for x in images], return_tensors='pt',do_rescale=False).pixel_values
    labels = torch.tensor(labels, dtype=torch.long)

    # Ensure weights are correctly aligned with the labels for the batch
    weights = torch.tensor([dataset.weights[label] for label in labels], dtype=torch.float)

    return {
        'pixel_values': pixel_values,
        'labels': labels,
        'weights': weights  # Make sure weights are used in the loss calculation
    }

# Metric
metric = load_metric("accuracy")

def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

# Training arguments
training_args = TrainingArguments(
    output_dir='./model_save',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    logging_dir='./logs',
    num_train_epochs=10,
    per_device_train_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    learning_rate=0.01,
    logging_strategy='epoch', 
    save_total_limit=3,  
    load_best_model_at_end=True,  
    metric_for_best_model='accuracy',  
    greater_is_better=True  
)

# Trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    callbacks=[metrics_callback]
)

# Train the model
trainer.train()
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(metrics_callback.train_loss_set, label='Training Loss')
plt.plot(metrics_callback.eval_loss_set, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(metrics_callback.eval_accuracy_set, label='Validation Accuracy')
plt.title('Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('./metrics_plot.png')
plt.show()

# Confusion Matrix
predictions, labels, _ = trainer.predict(eval_dataset)
predictions = np.argmax(predictions, axis=1)
cm = confusion_matrix(labels, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title('Confusion Matrix')
plt.savefig('./confusion_matrix.png')
plt.show()
model.save_pretrained('./classifier')
