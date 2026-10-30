from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class PatientInfo(BaseModel):
    patient_id: str
    name: str
    age: str
    gender: str

class PredictionRecord(BaseModel):
    patient_id: str
    patient_name: str
    patient_age: str
    patient_gender: str
    module: str  # "xray", "brain", or "skin"
    diagnosis: str
    risk_score: int
    probabilities: dict
    timestamp: datetime = Field(default_factory=datetime.utcnow)