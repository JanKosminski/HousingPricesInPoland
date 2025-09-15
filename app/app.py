import requests
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import xgboost as xgb


# Your training loader
def load_model_from_github(url):
    response = requests.get(url)
    response.raise_for_status()
    model_bytes = response.content

    model = xgb.XGBRegressor(enable_categorical=True, random_state=42)
    model.load_model(model_bytes)
    return model

# Load training at startup

model = load_model_from_github("https://github.com/JanKosminski/HousingPricesInPoland/blob/02602b3c9efcda6d5fba0efe907cae4f62193c60/trained_model/model_new_hyper.model")

# Define input schema
class PropertyData(BaseModel):
    city: str
    type: str
    squareMeters: float
    rooms: float
    floor: float = None
    floorCount: float = None
    buildYear: float = None
    latitude: float
    longitude: float
    centreDistance: float
    poiCount: float
    schoolDistance: float = None
    clinicDistance: float = None
    postOfficeDistance: float = None
    kindergartenDistance: float = None
    restaurantDistance: float = None
    collegeDistance: float = None
    pharmacyDistance: float = None
    ownership: str
    buildingMaterial: str = None
    condition: str = None
    hasParkingSpace: str
    hasBalcony: str
    hasElevator: str = None
    hasSecurity: str
    hasStorageRoom: str
    date: str  # "YYYY-MM-DD"

app = FastAPI()

@app.post("/predict")
def predict(data: PropertyData):
    # Convert input to DataFrame
    input_dict = data.model_dump()
    df = pd.DataFrame([input_dict])

    # Feature engineering
    df["month"] = pd.to_datetime(df["date"]).dt.month
    df["year"] = pd.to_datetime(df["date"]).dt.year
    df["pricePerSQM"] = df["price"] / df["squareMeters"]
    df = df.drop(columns=["date", "price"])
    df.drop(columns=["date"], inplace=True)

    # Fill missing values
    for col in df.columns:
        if df[col].isnull().any():
            df[col].fillna("Unknown" if df[col].dtype == "object" else df[col].median(), inplace=True)

    cat_columns = df.select_dtypes(include='object').columns
    df[cat_columns] = df[cat_columns].astype('category')

    # Predict
    prediction = model.predict(df).tolist()
    return {"prediction": prediction}
