from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load model + scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

app = FastAPI(title="Machine Failure Prediction API")

class MachineInput(BaseModel):
    rotational_speed: float
    torque: float
    tool_wear: float
    twf: int
    hdf: int
    pwf: int
    osf: int

@app.get("/")
def home():
    return {"message": "Machine Failure Prediction API is running!"}

@app.post("/predict")
def predict_failure(data: MachineInput):
    input_data = np.array([[
        data.rotational_speed,
        data.torque,
        data.tool_wear,
        data.twf,
        data.hdf,
        data.pwf,
        data.osf
    ]])

    input_scaled = scaler.transform(input_data)
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    return {
        "prediction": int(pred),
        "failure_probability": round(float(prob), 4)
    }
