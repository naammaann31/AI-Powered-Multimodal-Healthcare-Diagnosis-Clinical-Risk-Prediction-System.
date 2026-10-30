"""
Diabetes risk model training — CDC BRFSS 2015 Diabetes Health Indicators Dataset
(253,680 rows, 21 real health/lifestyle features, replaces the old 768-row Pima dataset)

Methodology:
- Stratified 70/15/15 train/val/test split (stratified because the target is imbalanced ~86/14)
- StandardScaler fit on train only (no leakage into val/test)
- Mini-batch training with class-weighted loss to handle imbalance
- ReduceLROnPlateau scheduler + early stopping on validation AUC (not accuracy, which is
  misleading on imbalanced data — a model predicting "no diabetes" always would score 86%)
- Final evaluation on held-out test set: accuracy, precision, recall, F1, ROC-AUC, confusion matrix
"""
import json
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                              f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

FEATURE_COLUMNS = [
    'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
    'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
    'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
    'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income'
]

df = pd.read_csv('data/vitals/diabetes_binary_health_indicators_BRFSS2015.csv')
X = df[FEATURE_COLUMNS].values
y = df['Diabetes_binary'].values.astype(int)

# 70/15/15 stratified split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

with open('models/vitals_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('models/vitals_feature_columns.json', 'w') as f:
    json.dump(FEATURE_COLUMNS, f)

def to_loader(X, y, batch_size=512, shuffle=False):
    ds = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

train_loader = to_loader(X_train, y_train, shuffle=True)
val_loader = to_loader(X_val, y_val)
test_loader = to_loader(X_test, y_test)

class VitalsModel(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Linear(32, 2)
        )
    def forward(self, x):
        return self.network(x)

model = VitalsModel(len(FEATURE_COLUMNS)).to(device)

# Class-weighted loss to counter the 86/14 imbalance
class_counts = np.bincount(y_train)
class_weights = torch.FloatTensor(len(y_train) / (2 * class_counts)).to(device)
print(f"Class weights: {class_weights.tolist()}")

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

def evaluate(loader):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            out = model(xb)
            probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            preds = out.argmax(dim=1).cpu().numpy()
            all_probs.extend(probs); all_preds.extend(preds); all_labels.extend(yb.numpy())
    auc = roc_auc_score(all_labels, all_probs)
    f1 = f1_score(all_labels, all_preds)
    return auc, f1, all_preds, all_labels

EPOCHS = 60
PATIENCE = 8
best_auc = 0
epochs_no_improve = 0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    val_auc, val_f1, _, _ = evaluate(val_loader)
    scheduler.step(val_auc)
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss/len(train_loader):.4f} | Val AUC: {val_auc:.4f} | Val F1: {val_f1:.4f}")

    if val_auc > best_auc:
        best_auc = val_auc
        epochs_no_improve = 0
        torch.save(model.state_dict(), 'models/vitals_model.pth')
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1} (no improvement for {PATIENCE} epochs)")
            break

print(f"\nBest validation AUC: {best_auc:.4f}")

# Final test-set evaluation with the best checkpoint
model.load_state_dict(torch.load('models/vitals_model.pth'))
test_auc, test_f1, test_preds, test_labels = evaluate(test_loader)
acc = accuracy_score(test_labels, test_preds)
prec = precision_score(test_labels, test_preds)
rec = recall_score(test_labels, test_preds)
cm = confusion_matrix(test_labels, test_preds)

print("\n=== Held-out Test Set Results ===")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1:        {test_f1:.4f}")
print(f"ROC-AUC:   {test_auc:.4f}")
print("Confusion Matrix:\n", cm)
print("\n", classification_report(test_labels, test_preds, target_names=['No Diabetes', 'Diabetes']))

metrics = {
    'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': test_f1, 'roc_auc': test_auc,
    'confusion_matrix': cm.tolist(), 'train_size': len(X_train), 'val_size': len(X_val), 'test_size': len(X_test)
}
with open('models/vitals_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("\nVitals model, scaler, feature columns, and metrics saved! ✅")
