import tkinter as tk
from tkinter import ttk
import requests

url = "http://localhost:8000/"

payload_template = {
    "city": "warszawa",
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




root = tk.Tk()
root.title("Housing price in Poland")

entries = {}


def create_widget(row, key, value):
    # Grabbing metadata for API
    api_response = requests.post(url + "fetchMeta")
    metadata = api_response.json()["metadata"]


    # pola z ograniczonym wyborem złapane z jsona
    if key in metadata.keys():
        var = tk.StringVar(value=value)
        options = metadata[key]
        widget = ttk.OptionMenu(root, var, value, *options)
        widget.grid(row=row, column=1, padx=5, pady=2)
        entries[key] = var
        return


    # liczby całkowite
    if isinstance(value, int):
        var = tk.StringVar(value=str(value))
        widget = tk.Spinbox(root, from_=0, to=9999, textvariable=var)
        widget.grid(row=row, column=1, padx=5, pady=2)
        entries[key] = var
        return

    # liczby zmiennoprzecinkowe
    if isinstance(value, float):
        var = tk.StringVar(value=str(value))
        widget = tk.Entry(root, textvariable=var)
        widget.grid(row=row, column=1, padx=5, pady=2)
        entries[key] = var
        return

    # fallback: zwykły tekst
    var = tk.StringVar(value=str(value))
    widget = tk.Entry(root, textvariable=var)
    widget.grid(row=row, column=1, padx=5, pady=2)
    entries[key] = var


# Generowanie formularza
for i, (key, value) in enumerate(payload_template.items()):
    tk.Label(root, text=key).grid(row=i, column=0, sticky="w", padx=5, pady=2)
    create_widget(i, key, value)



def send_request():
    # Zbieranie danych z formularza
    payload = {key: entries[key].get() for key in entries}

    # Konwersja typów (FastAPI wymaga poprawnych typów)
    for key, value in payload.items():
        if value.replace('.', '', 1).isdigit():
            payload[key] = float(value) if '.' in value else int(value)

    try:
        response = requests.post(url+"predict", json=payload)
        result = response.json()
        data = result["prediction"]
        output_label.config(text=f"Wynik modelu: {data}")
    except Exception as e:
        output_label.config(text=f"Błąd: {e}")


# Przycisk wysyłania
send_button = tk.Button(root, text="Wyślij do API", command=send_request)
send_button.grid(row=len(payload_template)+1, column=0, columnspan=2, pady=10)

# Pole na wynik
output_label = tk.Label(root, text="", font=("Arial", 12, "bold"))
output_label.grid(row=len(payload_template)+2, column=0, columnspan=2, pady=10)

root.mainloop()
