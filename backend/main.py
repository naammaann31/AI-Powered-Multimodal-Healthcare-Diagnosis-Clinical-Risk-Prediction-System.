from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from auth import router as auth_router
from fastapi.responses import Response
import json
import warnings
import transformers
warnings.filterwarnings('ignore')
transformers.logging.set_verbosity_error()

from predict import predict_xray, predict_vitals, predict_brain, predict_skin
from gradcam import generate_gradcam_xray, generate_gradcam_brain, generate_gradcam_skin
from report import generate_report
from database import predictions_collection
from crud import save_prediction, get_all_predictions, get_patient_history

def get_brain_guidance(prediction):
    prediction = prediction.lower()

    if prediction == "glioma":
        return "Consult a neurologist. Treatment may include surgery, radiation therapy, or chemotherapy depending on severity."

    elif prediction == "meningioma":
        return "Usually benign. Doctors may recommend regular monitoring or surgical removal if symptoms appear."

    elif prediction == "pituitary":
        return "Consult an endocrinologist. Treatment may include hormone therapy, medication, or surgery."

    elif prediction == "no tumor":
        return "No tumor detected. Maintain regular health checkups and a healthy lifestyle."

    else:
        return "Consult a medical professional for further evaluation."

def get_skin_guidance(prediction):
    prediction = prediction.lower()

    if prediction == "melanoma":
        return "Urgent: consult a dermatologist immediately. Melanoma can spread quickly if untreated; early biopsy and excision greatly improve outcomes."
    elif prediction == "basal cell carcinoma":
        return "Consult a dermatologist. Usually slow-growing and highly treatable with early removal, but should not be ignored."
    elif prediction == "actinic keratosis":
        return "Consult a dermatologist. This is a precancerous lesion — treatment (cryotherapy, topical therapy) can prevent progression."
    elif prediction == "benign keratosis":
        return "Generally harmless. Monitor for changes in size, shape, or color and mention it at your next checkup."
    elif prediction == "dermatofibroma":
        return "Usually a harmless benign growth. Removal is optional and only needed if it's bothersome."
    elif prediction == "melanocytic nevus":
        return "Likely a common mole. Keep an eye on it using the ABCDE rule (Asymmetry, Border, Color, Diameter, Evolving) and see a doctor if it changes."
    elif prediction == "vascular lesion":
        return "Usually benign (e.g. a blood vessel growth). Consult a dermatologist if it bleeds, grows, or changes."
    else:
        return "Consult a dermatologist for further evaluation."

app = FastAPI(title="AI Healthcare Diagnosis API")
app.include_router(auth_router, prefix="/auth")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "AI Healthcare Diagnosis API is running! ✅"}

@app.post("/predict-xray")
async def predict_xray_endpoint(image: UploadFile = File(...)):
    image_bytes = await image.read()
    result = predict_xray(image_bytes)
    return {"status": "success", **result}

@app.post("/predict-brain")
async def predict_brain_endpoint(image: UploadFile = File(...)):
    image_bytes = await image.read()
    result = predict_brain(image_bytes)

    if result.get("error"):
        return {"status": "error", "message": result["message"]}

    guidance = get_brain_guidance(result.get("diagnosis", ""))

    return {
        "status": "success",
        **result,
        "guidance": guidance
    }


@app.post("/predict-skin")
async def predict_skin_endpoint(image: UploadFile = File(...)):
    image_bytes = await image.read()
    result = predict_skin(image_bytes)

    if result.get("error"):
        return {"status": "error", "message": result["message"]}

    guidance = get_skin_guidance(result.get("diagnosis", ""))

    return {
        "status": "success",
        **result,
        "guidance": guidance
    }


@app.post("/predict-vitals")
async def predict_vitals_endpoint(
    high_bp: float = Form(...),
    high_chol: float = Form(...),
    chol_check: float = Form(...),
    bmi: float = Form(...),
    smoker: float = Form(...),
    stroke: float = Form(...),
    heart_disease_or_attack: float = Form(...),
    phys_activity: float = Form(...),
    fruits: float = Form(...),
    veggies: float = Form(...),
    hvy_alcohol_consump: float = Form(...),
    any_healthcare: float = Form(...),
    no_doc_bc_cost: float = Form(...),
    gen_hlth: float = Form(...),
    ment_hlth: float = Form(...),
    phys_hlth: float = Form(...),
    diff_walk: float = Form(...),
    sex: float = Form(...),
    age: float = Form(...),
    education: float = Form(...),
    income: float = Form(...)
):
    # Order must match models/vitals_feature_columns.json (training-time feature order)
    vitals = [high_bp, high_chol, chol_check, bmi, smoker, stroke,
              heart_disease_or_attack, phys_activity, fruits, veggies,
              hvy_alcohol_consump, any_healthcare, no_doc_bc_cost, gen_hlth,
              ment_hlth, phys_hlth, diff_walk, sex, age, education, income]
    result = predict_vitals(vitals)
    return {"status": "success", **result}

@app.post("/gradcam-xray")
async def gradcam_xray_endpoint(image: UploadFile = File(...)):
    image_bytes = await image.read()
    result = generate_gradcam_xray(image_bytes)
    if result.get("error"):
        return {"status": "error", "message": result["message"]}
    return {"status": "success", **result}

@app.post("/gradcam-brain")
async def gradcam_brain_endpoint(image: UploadFile = File(...)):
    image_bytes = await image.read()
    result = generate_gradcam_brain(image_bytes)
    if result.get("error"):
        return {"status": "error", "message": result["message"]}
    return {"status": "success", **result}

@app.post("/gradcam-skin")
async def gradcam_skin_endpoint(image: UploadFile = File(...)):
    image_bytes = await image.read()
    result = generate_gradcam_skin(image_bytes)
    if result.get("error"):
        return {"status": "error", "message": result["message"]}
    return {"status": "success", **result}

@app.post("/generate-report")
async def generate_report_endpoint(
    image: UploadFile = File(...),
    module: str = Form(...),
    patient_name: str = Form(''),
    patient_id: str = Form(''),
    patient_age: str = Form(''),
    patient_gender: str = Form('Male'),
    xray_diagnosis: str = Form(''),
    xray_risk_score: int = Form(0),
    xray_prob_normal: float = Form(0),
    xray_prob_pneumonia: float = Form(0),
    vitals_diagnosis: str = Form(''),
    vitals_risk_score: int = Form(0),
    vitals_prob_no_diabetes: float = Form(0),
    vitals_prob_diabetes: float = Form(0),
    brain_diagnosis: str = Form(''),
    brain_risk_score: int = Form(0),
    brain_prob_glioma: float = Form(0),
    brain_prob_meningioma: float = Form(0),
    brain_prob_no_tumor: float = Form(0),
    brain_prob_pituitary: float = Form(0),
    skin_diagnosis: str = Form(''),
    skin_risk_score: int = Form(0),
    skin_probabilities: str = Form(''),
    heatmap: str = Form('')
):
    image_bytes = await image.read()
    patient = {
        'name': patient_name, 'id': patient_id,
        'age': patient_age, 'gender': patient_gender
    }

    xray_result = None
    vitals_result = None
    brain_result = None
    skin_result = None

    if module == 'xray' and xray_diagnosis:
        xray_result = {
            'diagnosis': xray_diagnosis,
            'risk_score': xray_risk_score,
            'probabilities': {
                'Normal': xray_prob_normal,
                'Pneumonia': xray_prob_pneumonia
            }
        }
        vitals_result = {
            'diagnosis': vitals_diagnosis,
            'risk_score': vitals_risk_score,
            'probabilities': {
                'No Diabetes': vitals_prob_no_diabetes,
                'Diabetes': vitals_prob_diabetes
            }
        }

    if module == 'brain' and brain_diagnosis:
        brain_result = {
            'diagnosis': brain_diagnosis,
            'risk_score': brain_risk_score,
            'probabilities': {
                'Glioma': brain_prob_glioma,
                'Meningioma': brain_prob_meningioma,
                'No Tumor': brain_prob_no_tumor,
                'Pituitary': brain_prob_pituitary
            }
        }

    if module == 'skin' and skin_diagnosis:
        skin_result = {
            'diagnosis': skin_diagnosis,
            'risk_score': skin_risk_score,
            'probabilities': json.loads(skin_probabilities) if skin_probabilities else {}
        }

    pdf_bytes = generate_report(
        patient=patient,
        module=module,
        xray_result=xray_result,
        vitals_result=vitals_result,
        brain_result=brain_result,
        skin_result=skin_result,
        heatmap_b64=heatmap if heatmap else None,
        original_image_bytes=image_bytes
    )

    patient_name_clean = patient_name.replace(' ', '_') or 'Patient'
    filename = f"MediAI_Report_{patient_name_clean}.pdf"

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

# ─── Database & History Endpoints ───────────────────────────────────────────

@app.post("/save-prediction")
async def save_prediction_endpoint(data: dict):
    id = await save_prediction(data)
    return {"status": "success", "id": id}

@app.get("/history")
async def get_history():
    records = await get_all_predictions()
    return {"status": "success", "data": records}

@app.get("/history/{patient_id}")
async def get_patient_history_endpoint(patient_id: str):
    records = await get_patient_history(patient_id)
    return {"status": "success", "data": records}