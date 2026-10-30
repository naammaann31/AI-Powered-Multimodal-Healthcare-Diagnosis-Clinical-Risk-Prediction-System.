import torch
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image, ImageDraw
import io
import base64
import warnings
warnings.filterwarnings('ignore')

from model_loader import xray_model, brain_model, skin_model, device
from predict import _check_xray_like, _open_image, InvalidImageError, SKIN_CLASSES

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Skin lesion photos keep their color (see predict.py's transform_color) — no Grayscale.
transform_color = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        probs = torch.softmax(output, dim=1)[0]
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        loss = output[0, class_idx]
        loss.backward()
        gradients = self.gradients[0]
        activations = self.activations[0]
        weights = gradients.mean(dim=(1, 2))
        cam = torch.zeros(activations.shape[1:], device=device)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        cam = torch.relu(cam)
        cam = cam.cpu().numpy()
        cam = cv2.resize(cam, (224, 224))
        if cam.max() != cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        # Return probs too — no second forward pass needed!
        return cam, class_idx, probs

def apply_focused_heatmap(original_img, cam, label, confidence):
    original = np.array(original_img.resize((224, 224)))
    if len(original.shape) == 2:
        original = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
    elif original.shape[2] == 4:
        original = original[:, :, :3]
    original = original.copy()
    threshold = 0.70
    mask = cam > threshold
    overlay = original.copy()
    red_layer = np.zeros_like(original)
    red_layer[mask] = [220, 50, 50]
    alpha = 0.55
    overlay[mask] = cv2.addWeighted(
        original[mask].reshape(-1, 3), 1 - alpha,
        red_layer[mask].reshape(-1, 3), alpha, 0
    ).reshape(-1, 3)
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 80, 80), 2)
    pil_img = Image.fromarray(overlay)
    draw = ImageDraw.Draw(pil_img)
    draw.rectangle([0, 0, 224, 36], fill=(220, 30, 30))
    draw.text((8, 8), f"AI: {label}  ({confidence:.1f}%)", fill=(255, 255, 255))
    draw.rectangle([0, 196, 224, 224], fill=(20, 20, 20))
    draw.rectangle([6, 204, 20, 218], fill=(220, 50, 50))
    draw.text((24, 205), "High Risk Region", fill=(255, 255, 255))
    return np.array(pil_img)

def generate_gradcam_xray(image_bytes: bytes):
    try:
        img = _open_image(image_bytes)
    except InvalidImageError as e:
        return {"error": True, "message": str(e)}

    reject_reason = _check_xray_like(img)
    if reject_reason:
        return {"error": True, "message": reject_reason}

    tensor = transform(img).unsqueeze(0).to(device)
    tensor.requires_grad_(True)
    target_layer = xray_model.layer4[-1]
    gradcam = GradCAM(xray_model, target_layer)
    cam, pred_class, probs = gradcam.generate(tensor)
    classes = ['Normal', 'Pneumonia']
    label = classes[pred_class]
    confidence = probs[pred_class].item() * 100
    if pred_class == 1:
        overlay = apply_focused_heatmap(img, cam, label, confidence)
    else:
        original = np.array(img.resize((224, 224)))
        pil_img = Image.fromarray(original)
        draw = ImageDraw.Draw(pil_img)
        draw.rectangle([0, 0, 224, 36], fill=(22, 163, 74))
        draw.text((8, 8), f"AI: {label}  ({confidence:.1f}%)", fill=(255, 255, 255))
        draw.rectangle([0, 196, 224, 224], fill=(20, 20, 20))
        draw.rectangle([6, 204, 20, 218], fill=(22, 163, 74))
        draw.text((24, 205), "No Abnormality Detected", fill=(255, 255, 255))
        overlay = np.array(pil_img)
    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return {"heatmap": base64.b64encode(buffer).decode('utf-8'), "predicted_class": label}

def generate_gradcam_skin(image_bytes: bytes):
    try:
        img = _open_image(image_bytes)
    except InvalidImageError as e:
        return {"error": True, "message": str(e)}
    tensor = transform_color(img).unsqueeze(0).to(device)
    tensor.requires_grad_(True)
    target_layer = skin_model.features[-1]
    gradcam = GradCAM(skin_model, target_layer)
    cam, pred_class, probs = gradcam.generate(tensor)
    label = SKIN_CLASSES[pred_class]
    confidence = probs[pred_class].item() * 100
    NV_IDX = 5  # Melanocytic Nevus — common benign mole, treated like brain's "no tumor"
    if pred_class != NV_IDX:
        overlay = apply_focused_heatmap(img, cam, label, confidence)
    else:
        original = np.array(img.resize((224, 224)))
        pil_img = Image.fromarray(original)
        draw = ImageDraw.Draw(pil_img)
        draw.rectangle([0, 0, 224, 36], fill=(22, 163, 74))
        draw.text((8, 8), f"AI: {label}  ({confidence:.1f}%)", fill=(255, 255, 255))
        draw.rectangle([0, 196, 224, 224], fill=(20, 20, 20))
        draw.rectangle([6, 204, 20, 218], fill=(22, 163, 74))
        draw.text((24, 205), "Likely Benign Mole", fill=(255, 255, 255))
        overlay = np.array(pil_img)
    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return {"heatmap": base64.b64encode(buffer).decode('utf-8'), "predicted_class": label}

def generate_gradcam_brain(image_bytes: bytes):
    try:
        img = _open_image(image_bytes)
    except InvalidImageError as e:
        return {"error": True, "message": str(e)}
    tensor = transform(img).unsqueeze(0).to(device)
    tensor.requires_grad_(True)
    target_layer = brain_model.features[-1]
    gradcam = GradCAM(brain_model, target_layer)
    cam, pred_class, probs = gradcam.generate(tensor)
    classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
    label = classes[pred_class]
    confidence = probs[pred_class].item() * 100
    if pred_class != 2:
        overlay = apply_focused_heatmap(img, cam, label, confidence)
    else:
        original = np.array(img.resize((224, 224)))
        pil_img = Image.fromarray(original)
        draw = ImageDraw.Draw(pil_img)
        draw.rectangle([0, 0, 224, 36], fill=(22, 163, 74))
        draw.text((8, 8), f"AI: {label}  ({confidence:.1f}%)", fill=(255, 255, 255))
        draw.rectangle([0, 196, 224, 224], fill=(20, 20, 20))
        draw.rectangle([6, 204, 20, 218], fill=(22, 163, 74))
        draw.text((24, 205), "No Tumor Detected", fill=(255, 255, 255))
        overlay = np.array(pil_img)
    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return {"heatmap": base64.b64encode(buffer).decode('utf-8'), "predicted_class": label}