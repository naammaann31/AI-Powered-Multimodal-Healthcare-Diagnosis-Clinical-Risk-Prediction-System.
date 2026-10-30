"""
Chest X-ray pneumonia classification — RSNA Pneumonia Detection Challenge dataset
(26,684 images: 20,672 no-pneumonia / 6,012 pneumonia, radiologist-labeled).
Each image is one unique patient (no patient-level leakage risk like the old
Kermany dataset had), but there's no pre-made split, so we build one ourselves.

Methodology:
- Stratified 70/15/15 train/val/test split (fixed seed, reproducible)
- ResNet-50 pretrained on ImageNet, two-phase fine-tuning (frozen head warmup,
  then full unfreeze at low LR) — same approach as the brain MRI model
- Class-weighted loss for the ~3.4:1 class imbalance
- Mixed-precision training (AMP) for the 4GB VRAM budget
- Early stopping + ReduceLROnPlateau on validation ROC-AUC (imbalance-robust metric)
- 224x224 input to match the existing inference pipeline (predict.py / gradcam.py)
- Resumable: saves a full checkpoint (model/optimizer/scheduler/scaler + phase/epoch)
  after every epoch, so an interruption (sleep issue, forced restart, crash) only
  costs the in-progress epoch, not the whole run — just rerun this same script.
"""
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                              f1_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

DATA_ROOT = Path('data/images/Data')
CLASSES = ['nopneumonia', 'pneumonia']  # index 0 / 1
BEST_MODEL_PATH = 'models/image_model.pth'
CHECKPOINT_PATH = 'models/image_model_checkpoint.pth'

# Build full (filepath, label) list from the two class folders
all_files, all_labels = [], []
for class_idx, class_name in enumerate(CLASSES):
    for f in (DATA_ROOT / class_name).glob('*.jpg'):
        all_files.append(str(f))
        all_labels.append(class_idx)
all_files, all_labels = np.array(all_files), np.array(all_labels)
print(f"Total images: {len(all_files)} | Pneumonia: {all_labels.sum()} | No pneumonia: {(all_labels==0).sum()}")

# Stratified 70/15/15 split (fixed seed -> identical split across restarts)
train_f, temp_f, train_y, temp_y = train_test_split(all_files, all_labels, test_size=0.3, stratify=all_labels, random_state=42)
val_f, test_f, val_y, test_y = train_test_split(temp_f, temp_y, test_size=0.5, stratify=temp_y, random_state=42)
print(f"Train: {len(train_f)} | Val: {len(val_f)} | Test: {len(test_f)}")

class XrayDataset(Dataset):
    def __init__(self, files, labels, transform):
        self.files, self.labels, self.transform = files, labels, transform
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        return self.transform(img), self.labels[idx]

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

BATCH_SIZE = 24
train_loader = DataLoader(XrayDataset(train_f, train_y, train_transform), batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(XrayDataset(val_f, val_y, eval_transform), batch_size=BATCH_SIZE, num_workers=0, pin_memory=True)
test_loader = DataLoader(XrayDataset(test_f, test_y, eval_transform), batch_size=BATCH_SIZE, num_workers=0, pin_memory=True)

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
model = model.to(device)

class_counts = np.bincount(train_y)
class_weights = torch.FloatTensor(len(train_y) / (len(CLASSES) * class_counts)).to(device)
print(f"Class weights: {dict(zip(CLASSES, class_weights.tolist()))}")

criterion = nn.CrossEntropyLoss(weight=class_weights)
scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

def evaluate(loader):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                out = model(images)
            probs = torch.softmax(out.float(), dim=1)[:, 1].cpu().numpy()
            preds = out.argmax(dim=1).cpu().numpy()
            all_probs.extend(probs); all_preds.extend(preds); all_labels.extend(labels.numpy())
    auc = roc_auc_score(all_labels, all_probs)
    f1 = f1_score(all_labels, all_preds)
    return auc, f1, all_preds, all_labels

def save_checkpoint(phase, epochs_done, best_auc, epochs_no_improve, optimizer, scheduler):
    torch.save({
        'phase': phase,
        'epochs_done': epochs_done,
        'best_auc': best_auc,
        'epochs_no_improve': epochs_no_improve,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'scaler_state': scaler.state_dict(),
    }, CHECKPOINT_PATH)

def train_phase(phase_num, epochs, optimizer, scheduler, patience, best_auc_so_far,
                 start_epoch=0, epochs_no_improve_init=0):
    best_auc = best_auc_so_far
    epochs_no_improve = epochs_no_improve_init
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        n_batches = len(train_loader)
        print_every = max(1, n_batches // 20)  # ~20 progress lines per epoch
        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                out = model(images)
                loss = criterion(out, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            if (step + 1) % print_every == 0 or (step + 1) == n_batches:
                pct = 100 * (step + 1) / n_batches
                bar_filled = int(pct // 5)
                bar = '#' * bar_filled + '-' * (20 - bar_filled)
                print(f"  [{bar}] {pct:5.1f}% | batch {step+1}/{n_batches} | avg loss: {running_loss/(step+1):.4f}", flush=True)

        val_auc, val_f1, _, _ = evaluate(val_loader)
        scheduler.step(val_auc)
        print(f"  Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(train_loader):.4f} | Val AUC: {val_auc:.4f} | Val F1: {val_f1:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            epochs_no_improve = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
        else:
            epochs_no_improve += 1

        save_checkpoint(phase_num, epoch + 1, best_auc, epochs_no_improve, optimizer, scheduler)

        if epochs_no_improve >= patience:
            print(f"  Early stopping (no improvement for {patience} epochs)")
            break
    return best_auc

# ── Resume detection ──────────────────────────────────────────────────────
resume_phase, resume_epoch, resume_no_improve = 1, 0, 0
best_auc = 0

if os.path.exists(CHECKPOINT_PATH):
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    scaler.load_state_dict(ckpt['scaler_state'])
    resume_phase = ckpt['phase']
    resume_epoch = ckpt['epochs_done']
    resume_no_improve = ckpt['epochs_no_improve']
    best_auc = ckpt['best_auc']
    print(f"\nResuming from checkpoint: phase {resume_phase}, {resume_epoch} epoch(s) done, best AUC so far {best_auc:.4f}")
elif os.path.exists(BEST_MODEL_PATH):
    # A prior non-resumable run already finished phase 1 (best weights saved but no
    # mid-training checkpoint). Skip straight to phase 2 instead of redoing phase 1.
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    best_auc, _, _, _ = evaluate(val_loader)
    resume_phase, resume_epoch, resume_no_improve = 2, 0, 0
    print(f"\nFound existing best model (no checkpoint) — treating phase 1 as complete. "
          f"Re-validated AUC: {best_auc:.4f}. Skipping straight to phase 2.")

phase1_optimizer = phase1_scheduler = None
if resume_phase == 1:
    print("\n=== Phase 1: training classifier head (backbone frozen) ===")
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith('fc.')
    phase1_optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3, weight_decay=1e-4)
    phase1_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(phase1_optimizer, mode='max', factor=0.5, patience=2)
    if os.path.exists(CHECKPOINT_PATH) and resume_phase == 1:
        phase1_optimizer.load_state_dict(ckpt['optimizer_state'])
        phase1_scheduler.load_state_dict(ckpt['scheduler_state'])
    best_auc = train_phase(1, epochs=5, optimizer=phase1_optimizer, scheduler=phase1_scheduler,
                            patience=3, best_auc_so_far=best_auc,
                            start_epoch=resume_epoch, epochs_no_improve_init=resume_no_improve)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    resume_epoch, resume_no_improve = 0, 0

print("\n=== Phase 2: full fine-tuning (backbone unfrozen) ===")
for param in model.parameters():
    param.requires_grad = True
phase2_optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
phase2_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(phase2_optimizer, mode='max', factor=0.5, patience=2)
if os.path.exists(CHECKPOINT_PATH) and resume_phase == 2 and resume_epoch > 0:
    phase2_optimizer.load_state_dict(ckpt['optimizer_state'])
    phase2_scheduler.load_state_dict(ckpt['scheduler_state'])
best_auc = train_phase(2, epochs=15, optimizer=phase2_optimizer, scheduler=phase2_scheduler,
                        patience=5, best_auc_so_far=best_auc,
                        start_epoch=resume_epoch if resume_phase == 2 else 0,
                        epochs_no_improve_init=resume_no_improve if resume_phase == 2 else 0)

print(f"\nBest validation AUC: {best_auc:.4f}")

model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
test_auc, test_f1, test_preds, test_labels = evaluate(test_loader)
acc = accuracy_score(test_labels, test_preds)
cm = confusion_matrix(test_labels, test_preds)
report = classification_report(test_labels, test_preds, target_names=CLASSES, output_dict=True)

print("\n=== Held-out Test Set Results ===")
print(f"Accuracy: {acc:.4f}")
print(f"ROC-AUC:  {test_auc:.4f}")
print(f"F1:       {test_f1:.4f}")
print("Confusion Matrix:\n", cm)
print("\n", classification_report(test_labels, test_preds, target_names=CLASSES))

metrics = {
    'accuracy': acc, 'roc_auc': test_auc, 'f1': test_f1, 'confusion_matrix': cm.tolist(),
    'per_class_report': report, 'classes': CLASSES,
    'train_size': len(train_f), 'val_size': len(val_f), 'test_size': len(test_f)
}
with open('models/image_model_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

if os.path.exists(CHECKPOINT_PATH):
    os.remove(CHECKPOINT_PATH)

print("\nX-ray model and metrics saved! ✅")
