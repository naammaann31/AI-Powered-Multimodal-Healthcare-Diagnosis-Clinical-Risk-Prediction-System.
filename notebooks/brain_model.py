"""
Brain Tumor MRI classification — deduplicated dataset (6,507 images, 4 classes:
glioma, meningioma, notumor, pituitary). Pre-split train/val/test, no train/test
leakage (verified via pHash dedup by the dataset authors).

Methodology:
- EfficientNet-B3 pretrained on ImageNet, two-phase fine-tuning:
    Phase 1: backbone frozen, train classifier head only (fast convergence, avoids
             destroying pretrained features early with large random-init gradients)
    Phase 2: unfreeze entire network, fine-tune end-to-end at a lower LR
- Class-weighted loss (notumor class is ~35% smaller than the others)
- Mixed-precision training (AMP) — needed to fit comfortably in 4GB VRAM
- Early stopping + ReduceLROnPlateau on validation macro-F1 (not accuracy — macro-F1
  treats all 4 classes fairly despite the notumor class being smaller)
- 224x224 input to match the existing inference pipeline (predict.py / gradcam.py)
"""
import json

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

DATA_ROOT = 'data/brain_mri/brain-tumor-mri-deduplicated/data'
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']  # alphabetical, matches ImageFolder

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = ImageFolder(f'{DATA_ROOT}/train', transform=train_transform)
val_dataset = ImageFolder(f'{DATA_ROOT}/val', transform=eval_transform)
test_dataset = ImageFolder(f'{DATA_ROOT}/test', transform=eval_transform)
assert train_dataset.classes == CLASSES, f"Class order mismatch: {train_dataset.classes}"

print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
print(f"Classes: {train_dataset.classes}")

BATCH_SIZE = 16
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=0, pin_memory=True)

model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))
model = model.to(device)

# Class weights (inverse frequency) for the imbalanced notumor class
class_counts = np.bincount([label for _, label in train_dataset.samples])
class_weights = torch.FloatTensor(len(train_dataset) / (len(CLASSES) * class_counts)).to(device)
print(f"Class weights: {dict(zip(CLASSES, class_weights.tolist()))}")

criterion = nn.CrossEntropyLoss(weight=class_weights)
scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

def evaluate(loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                out = model(images)
            preds = out.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds); all_labels.extend(labels.numpy())
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    acc = accuracy_score(all_labels, all_preds)
    return macro_f1, acc, all_preds, all_labels

def train_phase(epochs, optimizer, scheduler, patience, best_f1_so_far):
    best_f1 = best_f1_so_far
    epochs_no_improve = 0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                out = model(images)
                loss = criterion(out, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        val_f1, val_acc, _, _ = evaluate(val_loader)
        scheduler.step(val_f1)
        print(f"  Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(train_loader):.4f} | Val macro-F1: {val_f1:.4f} | Val Acc: {val_acc:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'models/brain_tumor_model.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  Early stopping (no improvement for {patience} epochs)")
                break
    return best_f1

# Phase 1: freeze backbone, train classifier head only
print("\n=== Phase 1: training classifier head (backbone frozen) ===")
for param in model.features.parameters():
    param.requires_grad = False
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
best_f1 = train_phase(epochs=8, optimizer=optimizer, scheduler=scheduler, patience=4, best_f1_so_far=0)

# Phase 2: unfreeze everything, fine-tune end-to-end at lower LR
print("\n=== Phase 2: full fine-tuning (backbone unfrozen) ===")
for param in model.parameters():
    param.requires_grad = True
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
model.load_state_dict(torch.load('models/brain_tumor_model.pth'))  # resume from best phase-1 checkpoint
best_f1 = train_phase(epochs=20, optimizer=optimizer, scheduler=scheduler, patience=6, best_f1_so_far=best_f1)

print(f"\nBest validation macro-F1: {best_f1:.4f}")

# Final test-set evaluation
model.load_state_dict(torch.load('models/brain_tumor_model.pth'))
test_f1, test_acc, test_preds, test_labels = evaluate(test_loader)
cm = confusion_matrix(test_labels, test_preds)
report = classification_report(test_labels, test_preds, target_names=CLASSES, output_dict=True)

print("\n=== Held-out Test Set Results ===")
print(f"Accuracy: {test_acc:.4f}")
print(f"Macro-F1: {test_f1:.4f}")
print("Confusion Matrix:\n", cm)
print("\n", classification_report(test_labels, test_preds, target_names=CLASSES))

metrics = {
    'accuracy': test_acc, 'macro_f1': test_f1, 'confusion_matrix': cm.tolist(),
    'per_class_report': report, 'classes': CLASSES,
    'train_size': len(train_dataset), 'val_size': len(val_dataset), 'test_size': len(test_dataset)
}
with open('models/brain_tumor_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("\nBrain tumor model and metrics saved! ✅")
