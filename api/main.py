from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Real-Time AI Prediction Platform")

# Load trained model
model = joblib.load("models/model.pkl")


class HouseData(BaseModel):
    income: float
    house_age: float
    rooms: float
    bedrooms: float
    population: float


@app.get("/")
def home():
    return {"message": "AI Prediction Platform Running"}


@app.post("/predict")
def predict(data: HouseData):

    features = pd.DataFrame({
        "Avg. Area Income": [data.income],
        "Avg. Area House Age": [data.house_age],
        "Avg. Area Number of Rooms": [data.rooms],
        "Avg. Area Number of Bedrooms": [data.bedrooms],
        "Area Population": [data.population]
    })

    prediction = model.predict(features)[0]

    return {
        "predicted_price": round(float(prediction), 2)
    }