# Housing Prices in Poland

A data science project for predicting housing prices in Poland using machine learning (XGBoost) with data cleaning, model evaluation, and SHAP-based interpretability.

## Project Overview

This repository provides a simple pipeline for:
- Loading and cleaning apartment price data (from Kaggle)
- Training and tuning regression models to predict prices
- Evaluating model performance
- Interpreting predictions using SHAP values and visualizations

Predictions are driven by property size, location (city), construction year, and proximity to city center. Amenities like elevators, balconies, and proximity to points of interest are secondary features which refine model estimates.

## Repository Structure

- `main.py` — Main workflow: loads data, trains models, evaluates, and interprets.
- `data_handler.py` — Data acquisition (from Kaggle), cleaning, and splitting.
- `gridSearchVariant.py` — Alternative script focused on hyperparameter tuning for XGBoost - not used in code, proved to be sluggish, left to be re-used in future iterations. 
- `model_trainer.py` — Model training and tuning logic.
- `model_evaluation.py` — Model evaluation and visualization utilities.
- `misc.py` — Utility functions for robust data cleaning.
- `shap_visuals.py` — SHAP visualizations for feature impact analysis.
- `shap.md` — Documentation and short summary of SHAP findings.

## Installation

Clone the repository and install dependencies via pip:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is missing, install the following Python packages:

```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn kagglehub
```

## Usage

**Basic workflow:**

1. Ensure you have access to the Kaggle dataset: `krzysztofjamroz/apartment-prices-in-poland`.
2. Run the main script:

```bash
python main.py
```

- This will download the dataset, clean it, train a basic model for reference, start basic hyperparameter tuning to create better model, evaluate tuned performance, and display SHAP visualizations.


## Key Features

- **Data cleaning:** Handles missing values, standardizes categorical/binary columns, removes diacritics, deduplicates rows.
- **Modeling:** Uses XGBoost for regression; basic and tuned models supported.
- **Hyperparameter tuning:** Uses RandomizedSearchCV to tune the model.
- **Evaluation:** Prints metrics (MSE, MAE, R²), and plots actual vs. predicted prices with error coloring.
- **Interpretability:** SHAP analysis reveals key predictive features and visualizes their impact.
- **Extensible:** Modular scripts make it easy to extend with new models or data sources.

## Dependencies

- pandas
- numpy
- scikit-learn
- xgboost
- shap
- matplotlib
- seaborn
- kagglehub

## Data Source

- [Kaggle: Apartment Prices in Poland](https://www.kaggle.com/datasets/krzysztofjamroz/apartment-prices-in-poland)
  under Apache 2.0 licence

## Contributing

While this is just small personal project feel free fork the repository, submit pull requests, and to open issues for bugs.

## License

This project is licensed under the MIT License.  
You are free to use, modify, and distribute this code as you wish, provided you include the original license and attribution.
As I'm still learning about open source and best practices, I chose MIT for its simplicity and flexibility—if you notice anything I could improve about licensing or documentation, feel free to reach out or open a pull request!

## Contact

For questions or feedback, open an issue or contact [JanKosminski](https://github.com/JanKosminski).
