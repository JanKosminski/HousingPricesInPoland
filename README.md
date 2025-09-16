# Housing Prices in Poland

A data science project for predicting housing prices in Poland using machine learning (XGBoost) with data cleaning, model evaluation, and SHAP-based interpretability.

## Project Overview

This repository provides a simple pipeline for:
- Loading and cleaning apartment price data (from Kaggle)
- Training and tuning regression models to predict prices
- Evaluating model performance
- Interpreting predictions using SHAP values and visualizations

It also includes:
- A FastAPI-based API with an example request for prediction, enabling Docker integration
- A Dockerfile for containerized deployment
- A Trained model

Predictions are driven by property size, location (city), construction year, and proximity to city center. Amenities like elevators, balconies, and proximity to points of interest are secondary features which refine model estimates.

### Repository Structure

- training/
  - `main.py` — Main workflow: loads data, trains models, evaluates, and interprets.
  - `gridSearchVariant.py` — Alternative script focused on hyperparameter tuning for XGBoost - not used in code, proved to be sluggish, left to be re-used in future iterations. 
  - `model_trainer.py` — Model training and tuning logic.
  - `model_evaluation.py` — Model evaluation and visualization utilities.
  - `shap_visuals.py` — SHAP visualizations for feature impact analysis.
  - `shap.md` — Documentation and short summary of SHAP findings.
- data_processing/
  - `misc.py` — Utility functions for robust data cleaning.
  - `data_handler.py` — Data acquisition (from Kaggle), cleaning, and splitting.
- app/
  - `app.py` — FastAPI based API for Docker integration
  - `request_example.py` — An example of request one can make to the hosted model
- img/ - Charts related to model
- trained_model/ - As name suggest, model is located in that directory

### Model Training

The part of repository in subdirectories called "data_processing" and "training" were used to train XGBoost Regressor model it uses most basic hyperparameter tuning method using scikit-learn RandomizedSearchCV
below is the before and after comparison:

#### Model before hyperparameter tuning:

| Metric               | Value          |
|----------------------|----------------|
| Maximum squared error | 3458573.845 |
| Maximum absolute error| 1337.644  |
| R2                   |        0.857        |

#### Model after hyperparameter tuning:

| Metric               | Value          |
|----------------------|----------------|
| Maximum squared error | 2703179.696 |
| Maximum absolute error| 1143.402  |
| R2                   |        0.888        |


#### Tuned model accuracy:
<img src="/img/fig1_predictions_vs_actual_prices_tuned.png" width="400">

Overall accuracy of predictions while far from perfect seem to be mostly working, pink dots most likely depict unusual housing offers, which were either underpriced or vastly overpriced compared to median.

<img src="/img/fig2_shap_beespread.png" width="400" title="Beeswarm plot">

Parameters such as city, longitude, and latitude are dominant predictors, showing strong geographic influence on price.
Size and age being runner ups, squareMeters and buildYear also have clear, consistent effects—larger and newer apartments tend to be priced higher.
Distance to center: centreDistance has a negative impact when high (far from center), which aligns with real-world expectations.
Amenities - features like hasElevator, hasBalcony, and proximity to clinics or pharmacies have smaller but still measurable effects.

<img src="/img/fig3_shap_decisiontree.png" width="400" title="decision tree">

An example on how decisions were based. As expected one can see strong influence of buildYear, location and in as well as *some monthly variance* there might be some seasonal change worth investigating further. 

<img src="/img/fig4_shap_value_by_city.png" width="400" title="impact of cities">

Most important price factor, cities. Interestingly values don't seem balanced much. This indicates huge disproportion in
prices between the top 5 priciest cities and 9 others with Poznan being somewhat in the middle. Unsurprisingly all the top 5 
are the biggest cities in country, both population wise and in economic power. 

<img src="/img/fig5_shap_amenities.png" width="400" title="amenities">

While least impactful in terms of price it's quite interesting how important seems to be existence of elevator in the building,
considering that data included not only flats but also single detached houses etc. while parking space was not all that much important.
This data seems quite surprising considering Poland has quite high mechanical vehicle per capita ratio and cities in general lack parking spaces.


## FastAPI Integration

The project includes a FastAPI-based microservice that exposes the trained model via a RESTful API. This allows users to send property data and receive price predictions instantly.

### Features

- Endpoint: `/predict` accepts JSON payloads with property features
- Example request: Provided in `request_example.py`
- Docker support: Easily deployable using the included `Dockerfile`
- Model loading: Automatically loads the trained XGBoost model from `trained_model/`

### Running the API

To start the FastAPI server locally:

```bash
uvicorn app.app:app --reload
````

## Installation for Model Training

To set up the environment for training the housing price prediction model:

1. Clone the `training/` and `data_processing/` directories from the repository.

2. Install the required dependencies using pip:

```bash
pip install -r requirements.txt

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
