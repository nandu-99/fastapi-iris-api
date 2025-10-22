# api/main.py

from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import os


# Load model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "model.pkl")
model = pickle.load(open(model_path, "rb"))

# FastAPI instance
app = FastAPI(title="Iris Prediction API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Pydantic model for request validation
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Root endpoint
@app.get("/")
def root():
    return {"message": "FastAPI Iris ML API is running!"}

# Predict endpoint
@app.post("/predict")
def predict(data: IrisInput):
    features = np.array([[data.sepal_length, data.sepal_width,
                          data.petal_length, data.petal_width]])
    prediction = model.predict(features)[0]
    species = ["Setosa", "Versicolor", "Virginica"][prediction]
    return {"prediction": species}
