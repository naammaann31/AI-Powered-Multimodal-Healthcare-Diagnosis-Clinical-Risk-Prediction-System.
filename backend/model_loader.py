import json
import torch
import torch.nn as nn
from torchvision import models
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

print("Loading X-Ray model...")
xray_model = models.resnet50(weights=None)
xray_model.fc = nn.Linear(xray_model.fc.in_features, 2)
xray_model.load_state_dict(torch.load('../models/image_model.pth', map_location=device))
xray_model = xray_model.to(device)
xray_model.eval()
print("X-Ray model ready!")

with open('../models/vitals_feature_columns.json', 'r') as f:
    vitals_feature_columns = json.load(f)

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

print("Loading Vitals model...")
vitals_model = VitalsModel(len(vitals_feature_columns)).to(device)
vitals_model.load_state_dict(torch.load('../models/vitals_model.pth', map_location=device))
vitals_model.eval()
print("Vitals model ready!")

print("Loading scaler...")
with open('../models/vitals_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
print("Scaler ready!")

print("Loading Brain Tumor model...")
brain_model = models.efficientnet_b3(weights=None)
brain_model.classifier[1] = nn.Linear(brain_model.classifier[1].in_features, 4)
brain_model.load_state_dict(torch.load('../models/brain_tumor_model.pth', map_location=device))
brain_model = brain_model.to(device)
brain_model.eval()
print("Brain Tumor model ready!")

print("Loading Skin Lesion model...")
skin_model = models.efficientnet_b3(weights=None)
skin_model.classifier[1] = nn.Linear(skin_model.classifier[1].in_features, 7)
skin_model.load_state_dict(torch.load('../models/skin_model.pth', map_location=device))
skin_model = skin_model.to(device)
skin_model.eval()
print("Skin Lesion model ready!")
