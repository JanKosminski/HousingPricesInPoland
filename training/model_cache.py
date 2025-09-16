import xgboost as xgb


def load_model(filename):
    tuned_model = xgb.XGBRegressor(enable_categorical=True, random_state=42)
    tuned_model.load_model(filename)
    return tuned_model

def save_model(model, filename):
    try:
        model.save_model(filename)
        print("XGBoost training saved successfully.")
    except IOError as e:
        print(f"Failure to save file, due to: {e}")