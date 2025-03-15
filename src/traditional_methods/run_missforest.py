import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import (
    AdaBoostClassifier, AdaBoostRegressor,
    BaggingClassifier, BaggingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
    StackingClassifier, StackingRegressor,
    VotingClassifier, VotingRegressor
)
from missforest import MissForest
import os
import json
import matplotlib.pyplot as plt
import logging
from math import sqrt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("run_output.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

base_estimators = [
    ('rf', RandomForestClassifier()),
    ('et', ExtraTreesClassifier()),
]

CLASSIFIERS = [
    AdaBoostClassifier(), BaggingClassifier(), ExtraTreesClassifier(),
    GradientBoostingClassifier(), HistGradientBoostingClassifier(),
    RandomForestClassifier(), StackingClassifier(estimators=base_estimators),
    VotingClassifier(estimators=base_estimators)
]

REGRESSORS = [
    AdaBoostRegressor(), BaggingRegressor(), ExtraTreesRegressor(),
    GradientBoostingRegressor(), HistGradientBoostingRegressor(),
    RandomForestRegressor(), StackingRegressor(estimators=base_estimators),
    VotingRegressor(estimators=base_estimators)
]

# Mask the data at the given ratio
def mask_data(data, ratio, seed):
    np.random.seed(seed)
    mask = np.random.rand(*data.shape) < ratio
    masked_data = data.copy()
    masked_data[mask] = np.nan
    return masked_data

# MissForest with specified classifiers and regressors
def apply_missforest(data, classifier=None, regressor=None):
    mf = MissForest(clf=classifier, rgr=regressor)
    return mf.fit_transform(data)

# Ground truth comparsions with MSE, MAE, RMSE and R2
def evaluate_predictions(ground_truth, imputed_data):
    mse = mean_squared_error(ground_truth, imputed_data)
    mae = mean_absolute_error(ground_truth, imputed_data)
    rmse = sqrt(mse)
    r2 = r2_score(ground_truth, imputed_data)
    return {'mse': mse, 'mae': mae, 'rmse': rmse, 'r2': r2}

def run_experiments(data, masking_ratios, n_runs=10, save_path="results"):
    os.makedirs(save_path, exist_ok=True)

    results = []

    # Iterate over classifiers and regressors together
    for classifier, regressor in zip(CLASSIFIERS, REGRESSORS):
        for ratio in masking_ratios:
            for run in range(1, n_runs + 1):
                random_state = run  # Unique seed per iteration

                try:
                    logger.info(f"Running MissForest with classifier: {classifier.__class__.__name__}, regressor: {regressor.__class__.__name__}, ratio: {ratio}, run: {run}")

                    masked_data = mask_data(data, ratio, random_state)

                    # Apply MissForest for imputation and evaluate
                    imputed_data = apply_missforest(masked_data, classifier, regressor)

                    metrics = evaluate_predictions(data, imputed_data)
                    logger.info(f"Metrics: {metrics}")

                    results.append({
                        'masking_ratio': ratio,
                        'classifier': classifier.__class__.__name__,
                        'regressor': regressor.__class__.__name__,
                        'run': run,
                        'metrics': metrics
                    })
                except Exception as e:
                    logger.error(f"Error at ratio {ratio}, classifier {classifier.__class__.__name__}, regressor {regressor.__class__.__name__}, run {run}: {e}")

    # Save results
    results_file = os.path.join(save_path, "results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)

    return results

def plot_results(results, save_path="results"):
    df = pd.DataFrame(results)

    plt.figure(figsize=(12, 8))
    for (clf_name, rgr_name), group in df.groupby(['classifier', 'regressor']):
        mse_values = group.groupby('masking_ratio', group_keys=False)['metrics'].apply(lambda x: np.mean([m['mse'] for m in x]))
        plt.plot(mse_values.index, mse_values.values, marker='o', label=f"{clf_name} + {rgr_name}")

    plt.xlabel("Masking Ratio")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.title("Imputation Performance Across Masking Ratios")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.grid(True)

    plot_file = os.path.join(save_path, "mse_plot_all.png")
    plt.savefig(plot_file, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    data = pd.read_csv("largest_balanced_complete_protein_subset.csv")
    masking_ratios = np.linspace(0.1, 0.9, 9)

    logger.info("Starting experiments")
    results = run_experiments(data, masking_ratios)
    logger.info("Experiments completed")
    plot_results(results)
    logger.info("Plots saved successfully")
