import os
import data_handler
import model_cache
import model_evaluation
import model_trainer
import shap_visuals

FILENAME = "model_new_hyper.model"


# ------------------------ LOADING DATA ------------------------------
c_df = data_handler.load_data()
# Preview the combined DataFrame
# print("Combined DataFrame preview:")
# print(c_df.head())
# Checking for duplicates
# print(f"Number of duplicates: {c_df.duplicated().sum()}")
c_df.info()
c_df.describe()

# ---------------------------- DATA CLEANUP --------------------------
c_df = data_handler.data_cleanup(c_df)
print("------------------------------------")

# -------------------------------TWEAKS ------------------------------


# c_df = c_df.drop(columns=['latitude', 'longitude'])
# c_df = c_df.drop(columns=['city'])
# Model worked best with all of the above enabled.


# ---------------------------- DATA SPLIT ----------------------------
X_train, X_test, y_train, y_test = data_handler.data_split(c_df)

# --------------------------- BASIC MODEL ----------------------------
basic_model = model_trainer.basic_model(X_train, y_train)
basic_metrics = model_evaluation.evaluate_model(basic_model, X_test, y_test)
print("Metrics before tuning")
model_evaluation.print_metrics(basic_metrics)
print("------------------------------------")

# ---------------------------- TUNING / LOADING -----------------------

if os.path.isfile(FILENAME):
    tuned_model = model_cache.load_model(FILENAME)
else:
    print("No XGBoost model found, training new one.")
    print("-------------------------------------------")
    tuned_model = model_trainer.model_tuner(X_train, y_train)
    model_cache.save_model(tuned_model, FILENAME)

tuned_metrics = model_evaluation.evaluate_model(tuned_model, X_test, y_test)
print("Metrics after tuning")
model_evaluation.print_metrics(tuned_metrics)
model_evaluation.error_plot(tuned_metrics, y_test)
print("------------------------------------")

# -------------------------------- SHAP --------------------------------

# explainer = shap.TreeExplainer(tuned_model)
# shap_values = explainer.shap_values(X_test)
# shap.summary_plot(shap_values, X_test)
#
# expected_value = explainer.expected_value
# sample_indices = np.random.choice(X_test.shape[0], size=100, replace=False)
# X_sample = X_test.iloc[sample_indices]
# shap_values_sample = shap_values[sample_indices]
# shap.decision_plot(expected_value, shap_values_sample, X_test)

shap_visuals.shap_visuals(tuned_model, X_test)