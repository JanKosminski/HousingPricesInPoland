from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV


def basic_model(X_train, y_train):
    basic_model = xgb.XGBRegressor(enable_categorical=True, random_state=42)
    basic_model.fit(X_train, y_train)
    return basic_model


def model_tuner(X_train, y_train) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(enable_categorical=True, random_state=42)
    # Hyperparameter space
    param_dist = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 1, 5],
        'reg_alpha': [0, 0.1, 1],
        'reg_lambda': [1, 5, 10]
    }

    # Search setup
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=50,
        scoring='neg_mean_absolute_error',
        cv=5,
        verbose=1,
        n_jobs=-1
    )

    search.fit(X_train, y_train)
    improved_parameters = search.best_params_
    print("Best Parameters:", search.best_params_)

    tuned_model = xgb.XGBRegressor(**improved_parameters, enable_categorical=True, random_state=42)
    tuned_model.fit(X_train, y_train, verbose=True)

    return tuned_model
