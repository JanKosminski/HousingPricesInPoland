import os
import json
import xgboost as xgb


def load_model(filename):
    model_path = os.path.join(filename, "model_new_hyper.model")
    tuned_model = xgb.XGBRegressor(enable_categorical=True, random_state=42)
    tuned_model.load_model(model_path)
    return tuned_model

def save_model(model, directory, metadata):
    model_path = os.path.join(directory, "model_new_hyper.model")
    metadata_path = os.path.join(directory, "metadata.json")
    try:
        model.save_model(model_path)
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print("XGBoost training saved successfully.")
    except IOError as e:
        print(f"Failure to save file, due to: {e}")