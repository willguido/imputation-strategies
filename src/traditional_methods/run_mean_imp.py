import os
import pandas as pd
import logging
import numpy as np
import sys

# Add the root directory to sys path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
print("\n".join(sys.path))
print(f"Importing from: {os.path.abspath(os.path.dirname(__file__))}")

from traditional_methods.run_experiment import run_experiment
from utils.evaluation import plot_results
from utils.imputation import apply_mean_imputation

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mean_imputation_experiment_log.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

if __name__ == "__main__":
    logger.info("Starting mean imputation experiments")
    
    data_path = os.path.join("data", "largest_balanced_complete_protein_subset.csv")
    data = pd.read_csv(data_path)
    logger.info(f"Loaded data with shape: {data.shape}")

    masking_ratios = np.linspace(0.1, 0.9, 9)

    imputation_methods = {
        'Simple_Mean_Imputation': lambda data: apply_mean_imputation(data)
    }

    results = run_experiment(data, masking_ratios, imputation_methods, n_runs=7, results_path="experiments/results")
    logger.info("Mean imputation experiments completed")

    plot_results(
        results=results,
        metric="mse",
        title="Simple Mean Imputation Performance",
        save_path="experiments/results"
    )
    logger.info("Results plot saved")
