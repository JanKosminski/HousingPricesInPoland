import requests

url = "http://localhost:8000/predict"
payload = {
    "city": "Warsaw",
    "type": "apartment",
    "squareMeters": 75.0,
    "rooms": 3,
    "floor": 2,
    "floorCount": 5,
    "buildYear": 2010,
    "latitude": 52.2297,
    "longitude": 21.0122,
    "centreDistance": 3.5,
    "poiCount": 15,
    "schoolDistance": 0.8,
    "clinicDistance": 1.2,
    "postOfficeDistance": 0.5,
    "kindergartenDistance": 0.6,
    "restaurantDistance": 0.3,
    "collegeDistance": 2.0,
    "pharmacyDistance": 0.4,
    "ownership": "private",
    "buildingMaterial": "brick",
    "condition": "good",
    "hasParkingSpace": "yes",
    "hasBalcony": "yes",
    "hasElevator": "yes",
    "hasSecurity": "yes",
    "hasStorageRoom": "no",
    "date": "2023-09-15",
}

response = requests.post(url, json=payload)
print(response.json())