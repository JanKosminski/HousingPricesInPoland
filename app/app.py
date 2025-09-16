import tempfile
import requests
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import xgboost as xgb
from data_processing import misc

# Your training loader
def load_model_from_github(url):
    response = requests.get(url)
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name

    model = xgb.XGBRegressor(enable_categorical=True, random_state=42)
    model.load_model(tmp_path)
    return model


# Load training at startup

model = load_model_from_github("https://raw.githubusercontent.com/JanKosminski/HousingPricesInPoland/master/trained_model/model_new_hyper.model")

# Define input schema
class PropertyData(BaseModel):
    city: str
    type: str
    squareMeters: float
    rooms: float
    floor: float
    floorCount: float
    buildYear: float
    latitude: float
    longitude: float
    centreDistance: float
    poiCount: float
    schoolDistance: float
    clinicDistance: float
    postOfficeDistance: float
    kindergartenDistance: float
    restaurantDistance: float
    collegeDistance: float
    pharmacyDistance: float
    ownership: str
    buildingMaterial: str
    condition: str
    hasParkingSpace: str
    hasBalcony: str
    hasElevator: str
    hasSecurity: str
    hasStorageRoom: str
    date: str  # "YYYY-MM-DD"


app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the prediction API!"}

@app.post("/predict")
def predict(data: PropertyData):
    # Convert input to DataFrame
    input_dict = data.model_dump()
    df = pd.DataFrame([input_dict])

    # Feature engineering
    df["month"] = pd.to_datetime(df["date"]).dt.month
    df["year"] = pd.to_datetime(df["date"]).dt.year
    yes_or_no_columns = ['hasParkingSpace', 'hasBalcony', 'hasElevator', 'hasSecurity', 'hasStorageRoom']
    df = misc.validate_binary(yes_or_no_columns,df)
    cat_columns = df.select_dtypes(include='object').columns
    df[cat_columns] = df[cat_columns].astype('category')
    df = df.drop(columns=['date'])

    # Predict
    prediction = model.predict(df).tolist()
    return {"prediction": prediction}
