import kagglehub
import pandas as pd
import os
import re
from sklearn.model_selection import train_test_split
import misc


def load_data() -> pd.DataFrame:
    # Downloading most recent dataset
    path = kagglehub.dataset_download("krzysztofjamroz/apartment-prices-in-poland")
    print("Path to dataset files:", path)

    # Listing all CSV files in the folder
    csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
    # Load CSVs and add date column
    dataframes = []
    for file in csv_files:
        full_path = os.path.join(path, file)
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
    return c_df


def data_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    df = df.map(lambda x: misc.remove_diacritics(x.lower()) if isinstance(x, str) else x)
    df = df.drop_duplicates(subset="id", keep="last")
    df = df.drop(columns='id')
    yes_or_no_columns = ['hasParkingSpace', 'hasBalcony', 'hasElevator', 'hasSecurity', 'hasStorageRoom']
    df = misc.validate_binary(yes_or_no_columns, df)
    df = misc.fill_with_median(df)
    df = misc.fill_with_unknown(df)

    # ---- additional data for analysis
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['pricePerSQM'] = df['price']/df['squareMeters']
    df = df.drop(columns="price")

    # encode to categories
    cat_columns = df.select_dtypes(include='object').columns
    df[cat_columns] = df[cat_columns].astype('category')
    # df.info()
    # df.describe()
    return df


def data_split(df: pd.DataFrame):
    X = df.drop(columns=['pricePerSQM', 'date'])  # Features
    y = df['pricePerSQM']  # Target
    # First split: (train + val) vs test -> 80 : 20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test
