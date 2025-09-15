import shap
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def shap_visuals(model, X_test):
    ### reduced sample for mucho faster runs, anything below 3k looks bad on beeswarm
    X_sample = X_test.sample(n=5000, random_state=42)
    # shap_values = explainer.shap_values(X_test)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample, check_additivity=False)

    # beeswarm plot
    shap.summary_plot(shap_values, X_sample, plot_type="dot")

    # decision tree plot
    expected_value = explainer.expected_value
    sample_idx = np.random.choice(X_sample.shape[0], size=20, replace=False)
    shap.decision_plot(expected_value,shap_values[sample_idx, :],X_sample.iloc[sample_idx])

    city_idx = X_sample.columns.get_loc('city')
    city_shap = pd.DataFrame({
        'city': X_sample['city'].values,
        'shap_value': shap_values[:, city_idx]
    })
    city_mean = city_shap.groupby('city')['shap_value'].mean().sort_values()

    plt.figure(figsize=(10, 6))
    sns.barplot(x=city_mean.values, y=city_mean.index, palette="viridis", legend=False)
    plt.title("Average SHAP Value by City")
    plt.xlabel("Mean SHAP Value")
    plt.ylabel("City")
    plt.tight_layout()
    plt.show()

    has_things = ['hasElevator', 'hasBalcony', 'hasParkingSpace', 'hasSecurity', 'hasStorageRoom']
    has_things_shap = {
        feature: np.abs(shap_values[:, X_test.columns.get_loc(feature)]).mean()
        for feature in has_things
    }

    amenity_df = pd.DataFrame.from_dict(has_things_shap, orient='index', columns=['mean_abs_shap'])
    amenity_df = amenity_df.sort_values(by='mean_abs_shap', ascending=True)

    plt.figure(figsize=(8, 5))
    sns.barplot(x='mean_abs_shap', y=amenity_df.index, data=amenity_df, palette="magma")
    plt.title("SHAP Impact of Amenities")
    plt.xlabel("Mean Absolute SHAP Value")
    plt.ylabel("Amenity")
    plt.tight_layout()
    plt.show()
