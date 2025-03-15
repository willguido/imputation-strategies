import os
import pandas as pd
import numpy as np
import logging
import sys

# Add the root directory to sys path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
print("\n".join(sys.path))
print(f"Importing from: {os.path.abspath(os.path.dirname(__file__))}")

from traditional_methods.run_experiment import run_experiment
from utils.imputation import apply_knn_imputer
from utils.evaluation import plot_results

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("knn_experiment_log.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

if __name__ == "__main__":
    logger.info("Starting KNN experiments")
    
    data_path = os.path.join("data", "largest_balanced_complete_protein_subset.csv")
    data = pd.read_csv(data_path)
    logger.info(f"Loaded data with shape: {data.shape}")

    masking_ratios = np.linspace(0.1, 0.9, 9)

    # Define knn with different n_neighbors
    imputation_methods = {
        'KNN_3': lambda data: apply_knn_imputer(data, n_neighbors=3),
        'KNN_5': lambda data: apply_knn_imputer(data, n_neighbors=5),
        'KNN_7': lambda data: apply_knn_imputer(data, n_neighbors=7),
        'KNN_10': lambda data: apply_knn_imputer(data, n_neighbors=10),
        'KNN_15': lambda data: apply_knn_imputer(data, n_neighbors=15)
    }

    results = run_experiment(data, masking_ratios, imputation_methods, n_runs=5, results_path="experiments/results")
    logger.info("KNN experiments completed")

    plot_results(
        results=results,
        metric="mse",
        title="KNN Imputation Performance",
        save_path="experiments/results"
    )
    logger.info("Results plot saved")