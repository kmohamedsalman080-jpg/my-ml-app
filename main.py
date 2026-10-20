import os
from dotenv import load_dotenv
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# 1. Load Environment Variables from .env
load_dotenv()

APP_NAME = os.getenv("APP_NAME", "Server Overheat Prediction API")
THRESHOLD = float(os.getenv("OVERHEAT_THRESHOLD", 85.0))

app = FastAPI(title=APP_NAME)

# 2. Saved Model-a Load panrom (Unga file name: server_model.pkl)
model = joblib.load("server_model.pkl")


class ServerMetrics(BaseModel):
  cpu_usage: float
  fan_speed_low: int


@app.get("/")
def home():
  return {
      "message": "Server Overheat Prediction API is Live!",
      "app_name": APP_NAME,
      "threshold_used": THRESHOLD,
  }


@app.post("/predict")
def predict_overheat(data: ServerMetrics):
  features = np.array([[data.cpu_usage, data.fan_speed_low]])
  prediction = model.predict(features)[0]
  result = "DANGER: Overheat Detected!" if prediction == 1 else "Normal"

  return {
      "status_code": int(prediction),
      "prediction_result": result,
      "threshold_used": THRESHOLD,
  }