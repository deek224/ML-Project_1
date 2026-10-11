from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).parent

app = FastAPI(title="Student Placement Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and the exact column order used during training
model = joblib.load(BASE_DIR / "placement_model.joblib")
columns = joblib.load(BASE_DIR / "columns.joblib")

BRANCHES = ["CSE", "IT", "ECE", "Mechanical", "Civil", "Commerce"]


class StudentData(BaseModel):
    gender: Literal["male", "female"]
    branch: Literal["CSE", "IT", "ECE", "Mechanical", "Civil", "Commerce"]
    study_hours: float = Field(ge=0, le=12)
    attendance: float = Field(ge=0, le=100)
    sleep_hours: float = Field(ge=3, le=12)
    internet_usage: float = Field(ge=0, le=12)
    assignments_completed: int = Field(ge=0, le=50)
    previous_score: float = Field(ge=0, le=100)
    extracurricular: Literal["yes", "no"]
    exam_score: float = Field(ge=0, le=100)


@app.get("/")
def home():
    return FileResponse("templates/index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: StudentData):
    # Same encoding as the notebook
    row = {
        "gender": 1 if data.gender == "male" else 0,  # LabelEncoder: female=0, male=1
        "study_hours": data.study_hours,
        "attendance": data.attendance,
        "sleep_hours": data.sleep_hours,
        "internet_usage": data.internet_usage,
        "assignments_completed": data.assignments_completed,
        "previous_score": data.previous_score,
        "extracurricular": 1 if data.extracurricular == "yes" else 0,
        "exam_score": data.exam_score,
    }

    # One-hot branch columns are UPPERCASE in training (branch_CSE, branch_IT, ...)
    branch = data.branch.upper()
    for col in columns:
        if col.startswith("branch_"):
            row[col] = 1 if col == f"branch_{branch}" else 0

    df_row = pd.DataFrame([row]).reindex(columns=columns, fill_value=0)

    pred = int(model.predict(df_row)[0])
    proba = float(model.predict_proba(df_row)[0][1])  # probability of "Placed"

    return {
        "prediction": "Placed" if pred == 1 else "Not Placed",
        "placed_probability": round(proba, 4),
    }

