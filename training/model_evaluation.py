from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

def evaluate_model(model: xgb.XGBRegressor, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "mse": mean_squared_error(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
        "y_pred": y_pred
    }


def error_plot(metrics, y_test):
    errors = abs(y_test - metrics['y_pred'])
    plt.scatter(y_test, metrics['y_pred'], c=errors, cmap='coolwarm', alpha=0.3)
    plt.colorbar(label='Absolute Error')
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Predicted vs Actual Apartment Prices")
    plt.show()


def print_metrics(metrics: dict):
    print(f"Maximum squared error : {metrics['mse']}")
    print(f"Maximum absolute error: {metrics['mae']}")
    print(f"R2 : {metrics['r2']}")