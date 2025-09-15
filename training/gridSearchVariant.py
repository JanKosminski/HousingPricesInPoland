import kagglehub
import pandas as pd
import os
import re
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
import misc

# Downloading most recent dataset
path = kagglehub.dataset_download("krzysztofjamroz/apartment-prices-in-poland")
print("Path to dataset files:", path)
csv_folder = path

# Listing all CSV files in the folder
csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]

# Load CSVs and add date column
dataframes = []
for file in csv_files:
    full_path = os.path.join(csv_folder, file)
    df = pd.read_csv(full_path)

    # Extract date from FILENAME using regex
    match = re.search(r'(\d{4})_(\d{2})', file)
    if match:
        year, month = match.groups()
        df['date'] = pd.to_datetime(f"{year}-{month}-01")  # First day of the month
    # Filter out renting prices for now.
    if not ("rent" in file):
        dataframes.append(df)
        print(f"Loaded {file} with {len(df)} records and date {df['date'].iloc[0]}.")

# Merging into singular file. c_df as in Combined DataFrame
c_df = pd.concat(dataframes, ignore_index=True)


# # Preview the combined DataFrame
# print("Combined DataFrame preview:")
# print(c_df.head())
# # Checking for duplicates
# print(f"Number of duplicates: {c_df.duplicated().sum()}")
# c_df.info()
# c_df.describe()


# ----Data cleanup ----
c_df = c_df.map(lambda x: misc.remove_diacritics(x.lower()) if isinstance(x, str) else x)
c_df = c_df.drop_duplicates()
c_df = c_df.drop(columns='id')
# converting y/n to binary for XGBoost
yes_or_no_columns = ['hasParkingSpace', 'hasBalcony', 'hasElevator', 'hasSecurity', 'hasStorageRoom']
c_df = misc.validate_binary(yes_or_no_columns, c_df)

# filling missing numerical data with median
c_df = misc.fill_with_median(c_df)

# filling missing non-numericals with 'Unknown'
c_df = misc.fill_with_unknown(c_df)

# ---- additional data for analysis
c_df['month'] = c_df['date'].dt.month
c_df['year'] = c_df['date'].dt.year
# c_df['price_per_m2'] = c_df['price'] / c_df['squareMeters']

# encode to categories
cat_columns = c_df.select_dtypes(include='object').columns
c_df[cat_columns] = c_df[cat_columns].astype('category')

c_df.info()
c_df.describe()

# --------------------------- XGBOOST -------------------------------

X = c_df.drop(columns=['price', 'date'])  # Features
y = c_df['price']  # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBRegressor(enable_categorical=True)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Pre training")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R² Score: {r2:.4f}")



# Define the parameter grid
PARAMETERS = {
    "subsample": [0.5, 0.75, 1],
    "colsample_bytree": [0.5, 0.75, 1],
    "max_depth": [2, 6, 12],
    "min_child_weight": [1, 5, 15],
    "learning_rate": [0.3, 0.1, 0.03],
    "n_estimators": [100]
}

# Use XGBRegressor instead of XGBClassifier
model = xgb.XGBRegressor(n_jobs=-1, eval_metric='rmse', random_state=42, enable_categorical=True)

# Set up GridSearchCV with a regression scoring metric
model_gs = GridSearchCV(
    estimator=model,
    param_grid=PARAMETERS,
    cv=3,
    scoring="neg_mean_absolute_error",  # or "r2"
    verbose=1
)

# Fit the training
model_gs.fit(X_train, y_train)

# Show best parameters
print("Best Parameters:", model_gs.best_params_)
