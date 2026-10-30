"""
Skin lesion classification — HAM10000 dataset (10,015 dermatoscopic images, 7 classes:
akiec, bcc, bkl, df, mel, nv, vasc). No pre-made split, so we build one ourselves.
Images are split across two folders (HAM10000_images_part_1 / part_2); labels come
from HAM10000_metadata.csv (image_id -> dx).

Methodology:
- Stratified 70/15/15 train/val/test split (fixed seed, reproducible)
- EfficientNet-B3 pretrained on ImageNet, two-phase fine-tuning (frozen head warmup,
  then full unfreeze at low LR) — same approach as the brain MRI model
- Class-weighted loss — nv (melanocytic nevi) is ~67% of the dataset, df (dermatofibroma)
  is <1%, so this imbalance is far worse than the other two models and needs it
- Mixed-precision training (AMP) for the 4GB VRAM budget
- Early stopping + ReduceLROnPlateau on validation macro-F1 (imbalance-robust metric —
  plain accuracy would be dominated by the nv majority class)
- 224x224 input to match the existing inference pipeline (predict.py / gradcam.py)
- Resumable: saves a full checkpoint after every epoch, same as image_model.py

Speed: unlike the earlier two models, this one uses real multi-process data loading
(num_workers=4) to stop the single CPU thread from starving the GPU — that required
moving all run logic behind `if __name__ == '__main__':` since Windows spawns worker
processes by re-importing this file, which would otherwise recurse infinitely.

Safety: a background-free, inline GPU temperature check runs every progress print —
if the GPU is running hot it pauses training until it cools down, instead of racing
the laptop's thermals for the sake of speed.
"""
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

DATA_ROOT = Path('data/skin')
METADATA_CSV = DATA_ROOT / 'HAM10000_metadata.csv'
CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']  # alphabetical
BEST_MODEL_PATH = 'models/skin_model.pth'
CHECKPOINT_PATH = 'models/skin_model_checkpoint.pth'
BATCH_SIZE = 16
NUM_WORKERS = 4

GPU_PAUSE_ABOVE_C = 85   # pause training if the GPU hits this temperature
GPU_RESUME_BELOW_C = 78  # resume once it's cooled back down to this


def get_gpu_temp():
    try:
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        return int(out.stdout.strip().split('\n')[0])
    except Exception:
        return None


def gpu_safety_check():
    """Pause training if the GPU is running hot, resume once it cools down.
    A no-op (returns immediately) if nvidia-smi isn't available."""
    temp = get_gpu_temp()
    if temp is not None and temp >= GPU_PAUSE_ABOVE_C:
        print(f"\n  ⚠ GPU at {temp}°C — pausing training to cool down...", flush=True)
        while temp is not None and temp >= GPU_RESUME_BELOW_C:
            time.sleep(15)
            temp = get_gpu_temp()
        print(f"  GPU back to {temp}°C — resuming training.\n", flush=True)


class SkinDataset(Dataset):
    def __init__(self, files, labels, transform):
        self.files, self.labels, self.transform = files, labels, transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        return self.transform(img), self.labels[idx]


# NOTE: no Grayscale here — unlike X-ray/brain MRI, dermatoscopic images are
# genuinely color (pigmentation is a key diagnostic signal), so color is kept.
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        temp = get_gpu_temp()
        print(f"GPU starting temperature: {temp}°C" if temp is not None else "GPU temperature: unavailable (nvidia-smi not found)")

    # ── Build (filepath, label) list from metadata + the two image folders ─────
    meta = pd.read_csv(METADATA_CSV)
    image_dirs = [DATA_ROOT / 'HAM10000_images_part_1', DATA_ROOT / 'HAM10000_images_part_2']
    image_index = {}
    for d in image_dirs:
        if d.exists():
            for f in d.glob('*.jpg'):
                image_index[f.stem] = f

    class_to_idx = {c: i for i, c in enumerate(CLASSES)}
    all_files, all_labels = [], []
    missing = 0
    for _, row in meta.iterrows():
        img_id = row['image_id']
        dx = row['dx']
        if img_id in image_index and dx in class_to_idx:
            all_files.append(str(image_index[img_id]))
            all_labels.append(class_to_idx[dx])
        else:
            missing += 1
    all_files, all_labels = np.array(all_files), np.array(all_labels)
    print(f"Total images: {len(all_files)} (missing/unmatched: {missing})")
    print(f"Class counts: {dict(zip(CLASSES, np.bincount(all_labels, minlength=len(CLASSES)).tolist()))}")

    # Stratified 70/15/15 split (fixed seed -> identical split across restarts)
    train_f, temp_f, train_y, temp_y = train_test_split(all_files, all_labels, test_size=0.3, stratify=all_labels, random_state=42)
    val_f, test_f, val_y, test_y = train_test_split(temp_f, temp_y, test_size=0.5, stratify=temp_y, random_state=42)
    print(f"Train: {len(train_f)} | Val: {len(val_f)} | Test: {len(test_f)}")

    train_loader = DataLoader(SkinDataset(train_f, train_y, train_transform), batch_size=BATCH_SIZE,
                               shuffle=True, num_workers=NUM_WORKERS, pin_memory=True,
                               persistent_workers=True, prefetch_factor=4)
    val_loader = DataLoader(SkinDataset(val_f, val_y, eval_transform), batch_size=BATCH_SIZE,
                             num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(SkinDataset(test_f, test_y, eval_transform), batch_size=BATCH_SIZE,
                              num_workers=NUM_WORKERS, pin_memory=True)

    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))
    model = model.to(device)

    class_counts = np.bincount(train_y, minlength=len(CLASSES))
    class_weights = torch.FloatTensor(len(train_y) / (len(CLASSES) * class_counts)).to(device)
    print(f"Class weights: {dict(zip(CLASSES, class_weights.tolist()))}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    def evaluate(loader):
        model.eval()
        all_preds, all_labels_ = [], []
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                    out = model(images)
                preds = out.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds); all_labels_.extend(labels.numpy())
        macro_f1 = f1_score(all_labels_, all_preds, average='macro')
        acc = accuracy_score(all_labels_, all_preds)
        return macro_f1, acc, all_preds, all_labels_

    def save_checkpoint(phase, epochs_done, best_f1, epochs_no_improve, optimizer, scheduler):
        torch.save({
            'phase': phase,
            'epochs_done': epochs_done,
            'best_f1': best_f1,
            'epochs_no_improve': epochs_no_improve,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'scaler_state': scaler.state_dict(),
        }, CHECKPOINT_PATH)

    def train_phase(phase_num, epochs, optimizer, scheduler, patience, best_f1_so_far,
                     start_epoch=0, epochs_no_improve_init=0):
        best_f1 = best_f1_so_far
        epochs_no_improve = epochs_no_improve_init
        for epoch in range(start_epoch, epochs):
            model.train()
            running_loss = 0.0
            n_batches = len(train_loader)
            print_every = max(1, n_batches // 20)
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
                    gpu_temp = get_gpu_temp()
                    temp_str = f" | GPU {gpu_temp}°C" if gpu_temp is not None else ""
                    print(f"  [{bar}] {pct:5.1f}% | batch {step+1}/{n_batches} | avg loss: {running_loss/(step+1):.4f}{temp_str}", flush=True)
                    if device.type == 'cuda':
                        gpu_safety_check()

            val_f1, val_acc, _, _ = evaluate(val_loader)
            scheduler.step(val_f1)
            print(f"  Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(train_loader):.4f} | Val macro-F1: {val_f1:.4f} | Val Acc: {val_acc:.4f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                epochs_no_improve = 0
                torch.save(model.state_dict(), BEST_MODEL_PATH)
            else:
                epochs_no_improve += 1

            save_checkpoint(phase_num, epoch + 1, best_f1, epochs_no_improve, optimizer, scheduler)

            if epochs_no_improve >= patience:
                print(f"  Early stopping (no improvement for {patience} epochs)")
                break
        return best_f1

    # ── Resume detection ──────────────────────────────────────────────────
    resume_phase, resume_epoch, resume_no_improve = 1, 0, 0
    best_f1 = 0
    ckpt = None

    if os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        scaler.load_state_dict(ckpt['scaler_state'])
        resume_phase = ckpt['phase']
        resume_epoch = ckpt['epochs_done']
        resume_no_improve = ckpt['epochs_no_improve']
        best_f1 = ckpt['best_f1']
        print(f"\nResuming from checkpoint: phase {resume_phase}, {resume_epoch} epoch(s) done, best macro-F1 so far {best_f1:.4f}")
    elif os.path.exists(BEST_MODEL_PATH):
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
        best_f1, _, _, _ = evaluate(val_loader)
        resume_phase, resume_epoch, resume_no_improve = 2, 0, 0
        print(f"\nFound existing best model (no checkpoint) — treating phase 1 as complete. "
              f"Re-validated macro-F1: {best_f1:.4f}. Skipping straight to phase 2.")

    if resume_phase == 1:
        print("\n=== Phase 1: training classifier head (backbone frozen) ===")
        for param in model.features.parameters():
            param.requires_grad = False
        phase1_optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
        phase1_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(phase1_optimizer, mode='max', factor=0.5, patience=2)
        if ckpt is not None and resume_phase == 1:
            phase1_optimizer.load_state_dict(ckpt['optimizer_state'])
            phase1_scheduler.load_state_dict(ckpt['scheduler_state'])
        best_f1 = train_phase(1, epochs=8, optimizer=phase1_optimizer, scheduler=phase1_scheduler,
                               patience=4, best_f1_so_far=best_f1,
                               start_epoch=resume_epoch, epochs_no_improve_init=resume_no_improve)
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
        resume_epoch, resume_no_improve = 0, 0

    print("\n=== Phase 2: full fine-tuning (backbone unfrozen) ===")
    for param in model.parameters():
        param.requires_grad = True
    phase2_optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
    phase2_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(phase2_optimizer, mode='max', factor=0.5, patience=2)
    if ckpt is not None and resume_phase == 2 and resume_epoch > 0:
        phase2_optimizer.load_state_dict(ckpt['optimizer_state'])
        phase2_scheduler.load_state_dict(ckpt['scheduler_state'])
    best_f1 = train_phase(2, epochs=20, optimizer=phase2_optimizer, scheduler=phase2_scheduler,
                           patience=6, best_f1_so_far=best_f1,
                           start_epoch=resume_epoch if resume_phase == 2 else 0,
                           epochs_no_improve_init=resume_no_improve if resume_phase == 2 else 0)

    print(f"\nBest validation macro-F1: {best_f1:.4f}")

    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
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
        'train_size': len(train_f), 'val_size': len(val_f), 'test_size': len(test_f)
    }
    with open('models/skin_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)

    print("\nSkin cancer model and metrics saved! ✅")


if __name__ == '__main__':
    main()
