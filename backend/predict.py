import torch
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
import io
import numpy as np
import warnings
import transformers
warnings.filterwarnings('ignore')
transformers.logging.set_verbosity_error()

from model_loader import xray_model, vitals_model, brain_model, skin_model, scaler, vitals_feature_columns, device

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Skin lesion photos are genuinely color (pigmentation is a key diagnostic signal),
# unlike X-ray/brain MRI which are grayscale-by-modality — so no Grayscale here.
transform_color = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

SKIN_CLASSES = ['Actinic Keratosis', 'Basal Cell Carcinoma', 'Benign Keratosis',
                 'Dermatofibroma', 'Melanoma', 'Melanocytic Nevus', 'Vascular Lesion']
# Indices into SKIN_CLASSES considered malignant/precancerous (akiec, bcc, mel) —
# drives the risk score the same way brain's risk_score is (1 - P(no tumor)).
SKIN_MALIGNANT_IDX = [0, 1, 4]

# Heuristic confidence threshold below which a prediction is reported as "Uncertain"
# rather than forced into a class. Not statistically calibrated (would need a dedicated
# calibration set + reliability diagrams) — a reasonable, documented design choice so the
# system doesn't present a low-confidence guess as a confident diagnosis.
CONFIDENCE_THRESHOLD = 0.65


class InvalidImageError(Exception):
    """Raised when uploaded bytes can't be decoded as an image (wrong file type,
    empty upload, or corrupt/truncated image data)."""
    pass


def _open_image(image_bytes: bytes) -> Image.Image:
    """Decode uploaded bytes into a PIL Image, raising InvalidImageError for anything
    that isn't a readable image (instead of letting PIL's decode exceptions propagate
    as an unhandled 500)."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.load()  # force full decode now so truncated/corrupt data fails here, not later
        return img.convert('RGB')
    except (UnidentifiedImageError, OSError) as e:
        raise InvalidImageError(
            "The uploaded file isn't a valid image. Please upload a JPG or PNG image."
        ) from e


def _uncertain_result(diagnosis_label, risk_score, probabilities):
    return {
        "diagnosis": "Uncertain",
        "risk_score": risk_score,
        "probabilities": probabilities,
        "low_confidence": True,
        "note": f"Model confidence was below {int(CONFIDENCE_THRESHOLD*100)}%. "
                f"Closest match: {diagnosis_label}. Recommend clinical review."
    }


def _check_xray_like(img: Image.Image):
    """Reject inputs that don't statistically resemble a grayscale chest X-ray."""
    img_array = np.array(img)

    r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
    rg_diff = np.mean(np.abs(r.astype(int) - g.astype(int)))
    rb_diff = np.mean(np.abs(r.astype(int) - b.astype(int)))
    gb_diff = np.mean(np.abs(g.astype(int) - b.astype(int)))
    avg_color_diff = (rg_diff + rb_diff + gb_diff) / 3
    if avg_color_diff > 20:
        return "This doesn't look like a grayscale X-ray image (too much color). Please upload a chest X-ray."

    h, w = img_array.shape[:2]
    aspect_ratio = w / h
    if aspect_ratio < 0.5 or aspect_ratio > 2.0:
        return "Image proportions don't match a typical chest X-ray. Please upload a valid chest X-ray."

    if np.std(img_array) < 10:
        return "Image appears blank or near-uniform. Please upload a valid chest X-ray."

    return None


def predict_xray(image_bytes: bytes):
    try:
        img = _open_image(image_bytes)
    except InvalidImageError as e:
        return {"error": True, "message": str(e)}

    reject_reason = _check_xray_like(img)
    if reject_reason:
        return {"error": True, "message": reject_reason}

    tensor = transform(img).unsqueeze(0).to(device)
    with torch.inference_mode():
        output = xray_model(tensor)
        probs = torch.softmax(output, dim=1)[0]
    classes = ['Normal', 'Pneumonia']
    pred = torch.argmax(probs).item()
    confidence = probs[pred].item()

    probabilities = {
        "Normal":    round(probs[0].item() * 100, 2),
        "Pneumonia": round(probs[1].item() * 100, 2)
    }
    risk_score = int(probs[1].item() * 100)

    if confidence < CONFIDENCE_THRESHOLD:
        return {"error": False, **_uncertain_result(classes[pred], risk_score, probabilities)}

    return {
        "error": False,
        "diagnosis": classes[pred],
        "risk_score": risk_score,
        "probabilities": probabilities,
        "low_confidence": False
    }


def predict_vitals(vitals: list):
    """vitals must be ordered to match model_loader.vitals_feature_columns."""
    vitals_scaled = scaler.transform([vitals])
    tensor = torch.FloatTensor(vitals_scaled).to(device)
    with torch.inference_mode():
        output = vitals_model(tensor)
        probs = torch.softmax(output, dim=1)[0]
    classes = ['No Diabetes', 'Diabetes']
    pred = torch.argmax(probs).item()
    confidence = probs[pred].item()

    probabilities = {
        "No Diabetes": round(probs[0].item() * 100, 2),
        "Diabetes":    round(probs[1].item() * 100, 2)
    }
    risk_score = int(probs[1].item() * 100)

    if confidence < CONFIDENCE_THRESHOLD:
        return _uncertain_result(classes[pred], risk_score, probabilities)

    return {
        "diagnosis": classes[pred],
        "risk_score": risk_score,
        "probabilities": probabilities,
        "low_confidence": False
    }


def predict_brain(image_bytes: bytes):
    try:
        img = _open_image(image_bytes)
    except InvalidImageError as e:
        return {"error": True, "message": str(e)}
    img_array = np.array(img)

    r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
    rg_diff = np.mean(np.abs(r.astype(int) - g.astype(int)))
    rb_diff = np.mean(np.abs(r.astype(int) - b.astype(int)))
    gb_diff = np.mean(np.abs(g.astype(int) - b.astype(int)))
    avg_color_diff = (rg_diff + rb_diff + gb_diff) / 3

    if avg_color_diff > 20:
        return {"error": True, "message": "Invalid image! Please upload a Brain MRI scan only."}

    h, w = img_array.shape[:2]
    aspect_ratio = w / h
    if aspect_ratio < 0.7 or aspect_ratio > 1.5:
        return {"error": True, "message": "Image format doesn't match a Brain MRI scan."}

    overall_brightness = np.mean(img_array)
    if overall_brightness > 160:
        return {"error": True, "message": "This looks like a Chest X-Ray, not a Brain MRI."}

    tensor = transform(img).unsqueeze(0).to(device)
    with torch.inference_mode():
        output = brain_model(tensor)
        probs = torch.softmax(output, dim=1)[0]

    max_prob = probs.max().item()
    if max_prob < 0.5:
        return {"error": True, "message": "Image doesn't look like a Brain MRI scan."}

    classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
    pred = torch.argmax(probs).item()
    risk_score = int((1 - probs[2].item()) * 100)
    probabilities = {
        "Glioma":     round(probs[0].item() * 100, 2),
        "Meningioma": round(probs[1].item() * 100, 2),
        "No Tumor":   round(probs[2].item() * 100, 2),
        "Pituitary":  round(probs[3].item() * 100, 2)
    }

    if max_prob < CONFIDENCE_THRESHOLD:
        return {"error": False, **_uncertain_result(classes[pred], risk_score, probabilities)}

    return {
        "error": False,
        "diagnosis": classes[pred],
        "risk_score": risk_score,
        "probabilities": probabilities,
        "low_confidence": False
    }


def predict_skin(image_bytes: bytes):
    try:
        img = _open_image(image_bytes)
    except InvalidImageError as e:
        return {"error": True, "message": str(e)}

    img_array = np.array(img)
    h, w = img_array.shape[:2]
    aspect_ratio = w / h
    if aspect_ratio < 0.5 or aspect_ratio > 2.0:
        return {"error": True, "message": "Image proportions don't match a typical skin lesion photo."}

    if np.std(img_array) < 10:
        return {"error": True, "message": "Image appears blank or near-uniform. Please upload a valid skin lesion photo."}

    tensor = transform_color(img).unsqueeze(0).to(device)
    with torch.inference_mode():
        output = skin_model(tensor)
        probs = torch.softmax(output, dim=1)[0]

    pred = torch.argmax(probs).item()
    confidence = probs[pred].item()
    risk_score = int(sum(probs[i].item() for i in SKIN_MALIGNANT_IDX) * 100)
    probabilities = {SKIN_CLASSES[i]: round(probs[i].item() * 100, 2) for i in range(len(SKIN_CLASSES))}

    if confidence < CONFIDENCE_THRESHOLD:
        return {"error": False, **_uncertain_result(SKIN_CLASSES[pred], risk_score, probabilities)}

    return {
        "error": False,
        "diagnosis": SKIN_CLASSES[pred],
        "risk_score": risk_score,
        "probabilities": probabilities,
        "low_confidence": False
    }
