from fastapi import FastAPI
from pydantic import BaseModel

import sys
sys.path.append("backend")
from predictor import predict_risk

app = FastAPI(
    title="Pregnancy Risk API",
    description="Predicts maternal health risk level based on vital signs.",
    version="0.1.0"
)

class InputData(BaseModel):
    Age: float
    SystolicBP: float
    DiastolicBP: float
    BS: float
    BodyTemp: float
    HeartRate: float

@app.get("/")
def read_root():
    return {"message": "✅ Pregnancy Risk API is running."}

@app.post("/predict")
def predict(data: InputData):
    print("📡 /predict endpoint was triggered!")
    try:
        result = predict_risk(data.dict())
        return {"risk_level": result}
    except Exception as e:
        print(f"❌ Error in /predict: {e}")
        return {"error": str(e)}