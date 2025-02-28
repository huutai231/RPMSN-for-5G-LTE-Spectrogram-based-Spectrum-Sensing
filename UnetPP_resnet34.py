import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt

class SemanticSegmentationDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_paths = sorted([os.path.join(image_dir, img) for img in os.listdir(image_dir)])
        self.label_paths = sorted([os.path.join(label_dir, lbl) for lbl in os.listdir(label_dir)])

        # Define the RGB colors for each class
        self.class_colors = {
            (2, 0, 0): 0,       # LTE class
            (127, 0, 0): 1,     # RF class
            (248, 163, 191): 2  # Noise class
        }
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load label
        label = cv2.imread(self.label_paths[idx])
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)

        # Map RGB colors to class indices
        label_mask = np.zeros(label.shape[:2], dtype=np.uint8)
        for rgb, idx in self.class_colors.items():
            label_mask[np.all(label == rgb, axis=-1)] = idx

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)
            label_mask = torch.from_numpy(label_mask).long()

        return image, label_mask

# Usage example:
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),  # Resize to desired input size
    transforms.ToTensor()
])

train_dataset = SemanticSegmentationDataset(
    image_dir='/kaggle/input/spectogram-dataset/SpectrogramData/trainSet/input',
    label_dir='/kaggle/input/spectogram-dataset/SpectrogramData/trainSet/label',
    transform=train_transform
)

val_dataset = SemanticSegmentationDataset(
    image_dir='/kaggle/input/spectogram-dataset/SpectrogramData/testSet/input',
    label_dir='/kaggle/input/spectogram-dataset/SpectrogramData/testSet/label',
    transform=train_transform
)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

print(len(train_dataset))
print(len(val_dataset))
print(len(train_dataloader))
print(len(val_dataloader))
print(len(test_dataloader))

import torch
import segmentation_models_pytorch as smp
from torch.optim import Adam
import torch.nn as nn
from torchvision import models
import torch.optim as optim

model = smp.UnetPlusPlus(
    encoder_name="resnet34",  # Backbone của encoder
    encoder_weights=None,     # Don't use pretrained weights
    classes=3,                # Number of classes for segmentation
    activation="softmax"      
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = nn.DataParallel(model)
model.to(device)
print(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

total_params = count_parameters(model)
print(f"Total parameters: {total_params}")

from tqdm import tqdm
import torch.nn.functional as F
from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex
from torchmetrics import ConfusionMatrix

def train_epoch(model, dataloader, criterion, optimizer, device, num_classes):
    model.train()
    running_loss = 0.0
    
    confmat = ConfusionMatrix(task="multiclass", num_classes=num_classes).to(device)
    iou_score_avg = 0.0
    f1_score_avg = 0.0
    recall_avg = 0.0
    
    pbar = tqdm(dataloader, desc='Training', unit='batch')
    
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        
        preds = torch.argmax(outputs, dim=1)
        
        # Cập nhật confusion matrix
        confmat(preds, labels)
        
        tp, fp, fn, tn = smp.metrics.get_stats(preds, labels, mode='multiclass', num_classes=num_classes)
         # Tính các metric
        iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro")
        f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="macro")
        accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
        recall = smp.metrics.recall(tp, fp, fn, tn, reduction="macro")
        
        iou_score_avg += iou_score
        f1_score_avg += f1_score
        recall_avg += recall
        
        # Update tqdm with all metrics for each batch
        pbar.set_postfix({
            'Batch Loss': f'{loss.item():.4f}',
            'Accuracy': f'{accuracy:.4f}',
            'mIoU': f'{iou_score:.4f}',
            'F1 Score': f'{f1_score:.4f}',
            'Recall': f'{recall:.4f}'
        })
    
    cm = confmat.compute().cpu().numpy()  
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  
    confmat.reset()
    true_positive = np.diag(cm_normalized)
    total_samples_per_class = cm_normalized.sum(axis=1)
    mean_accuracy = np.mean(true_positive / total_samples_per_class)

    epoch_loss = running_loss / len(dataloader.dataset)
    iou_score_avg /= len(dataloader)
    f1_score_avg /= len(dataloader)
    recall_avg /= len(dataloader)
    
    return cm_normalized, epoch_loss, iou_score_avg, f1_score_avg, mean_accuracy, recall_avg

def evaluate(model, dataloader, criterion, device, num_classes):
    model.eval()
    running_loss = 0.0
    iou_score_avg = 0.0
    f1_score_avg = 0.0
    recall_avg = 0.0
    confmat = ConfusionMatrix(task="multiclass", num_classes=num_classes).to(device)

    pbar = tqdm(dataloader, desc='Evaluating', unit='batch')
    
    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            
            preds = torch.argmax(outputs, dim=1)

            confmat(preds, labels)
            
            tp, fp, fn, tn = smp.metrics.get_stats(preds, labels, mode='multiclass', num_classes=num_classes)
            iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro")
            f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="macro")
            accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
            recall = smp.metrics.recall(tp, fp, fn, tn, reduction="macro")

            iou_score_avg += iou_score
            f1_score_avg += f1_score
            recall_avg += recall
   
            pbar.set_postfix({
                'Batch Loss': f'{loss.item():.4f}',
                'Accuracy': f'{accuracy:.4f}',
                'mIoU': f'{iou_score:.4f}',
                'F1 Score': f'{f1_score:.4f}',
                'Recall': f'{recall:.4f}'
            })
    
    epoch_loss = running_loss / len(dataloader.dataset)
    cm = confmat.compute().cpu().numpy()  
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  
    confmat.reset()
    true_positive = np.diag(cm_normalized)
    total_samples_per_class = cm_normalized.sum(axis=1)
    mean_accuracy = np.mean(true_positive / total_samples_per_class)
    
    iou_score_avg /= len(dataloader)
    f1_score_avg /= len(dataloader)
    recall_avg /= len(dataloader)
    
    return cm_normalized, epoch_loss, iou_score_avg, f1_score_avg, mean_accuracy, recall_avg

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-5)

# Training model
num_epochs = 100
num_classes = 3
epoch_saved = 0

best_val_accuracy = 0.0

for epoch in range(num_epochs):
    _, epoch_loss_train, iou_score_avg_train, f1_score_avg_train, accuracy_avg_train, recall_avg_train = train_epoch(model, train_dataloader, criterion, optimizer, device, num_classes)
    _, epoch_loss_val, iou_score_avg_val, f1_score_avg_val, accuracy_avg_val, recall_avg_val = evaluate(model, val_dataloader, criterion, device, num_classes)
    
    print(f"Epoch {epoch + 1}/{num_epochs}")
    print(f"Train Loss: {epoch_loss_train:.4f}, Mean Accuracy: {accuracy_avg_train:.4f}, mIoU: {iou_score_avg_train:.4f}, F1 Score: {f1_score_avg_train:.4f}, Recall: {recall_avg_train:.4f}")
    print(f"Validation Loss: {epoch_loss_val:.4f}, Mean Accuracy: {accuracy_avg_val:.4f}, mIoU: {iou_score_avg_val:.4f}, F1 Score: {f1_score_avg_val:.4f}, Recall: {recall_avg_val:.4f}")

    if accuracy_avg_val >= best_val_accuracy:
        epoch_saved = epoch + 1
        best_val_accuracy = accuracy_avg_val

print("===================")
print(f"Best Model at epoch : {epoch_saved}")

model.load_state_dict(torch.load("/kaggle/working/UnetPlusPlus_ResNet34.pth"))

import torch
import matplotlib.pyplot as plt
import numpy as np

for i in range(100):
    print(i)
    image, label = val_dataset[i]

    
    image_np = image.permute(1, 2, 0).numpy()  # Convert from (C, H, W) to (H, W, C)
    image_np = (image_np * 255).astype(np.uint8)  # convert from [0, 1] to [0, 255]


    image_tensor = image.unsqueeze(0).to(device)

    # Predict with model
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)

    predict = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    label_colored = np.zeros_like(image_np)
    predict_colored = np.zeros_like(image_np)
    for rgb, idx in val_dataset.class_colors.items():
        label_colored[label.numpy() == idx] = rgb
        predict_colored[predict == idx] = rgb

    fig, axes = plt.subplots(1, 3, figsize=(20, 10))

    axes[0].imshow(image_np)
    axes[0].set_title("Image")
    axes[0].axis('off')

    axes[1].imshow(label_colored)
    axes[1].set_title("Label")
    axes[1].axis('off')

    axes[2].imshow(predict_colored)
    axes[2].set_title("Predict")
    axes[2].axis('off')

    plt.show()