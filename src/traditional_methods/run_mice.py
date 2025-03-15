import os
import pandas as pd
import logging
import sys

# Add the root directory to sys path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
print("\n".join(sys.path))
print(f"Importing from: {os.path.abspath(os.path.dirname(__file__))}")

from traditional_methods.run_experiment import run_experiment
from utils.imputation import apply_mice
from utils.evaluation import plot_results

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mice_experiment_log.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

if __name__ == "__main__":
    logger.info("Starting MICE experiments")
    
    data_path = os.path.join("data", "largest_balanced_complete_protein_subset.csv")
    data = pd.read_csv(data_path)
    logger.info(f"Loaded data with shape: {data.shape}")

    masking_ratios = [0.1, 0.3, 0.5, 0.7]

    # MICE with different estimators
    from sklearn.linear_model import BayesianRidge, Lasso, ElasticNet
    from sklearn.ensemble import ExtraTreesRegressor

    imputation_methods = {
        'MICE_BayesianRidge': lambda data: apply_mice(data, estimator=BayesianRidge()),
        "MICE_Lasso": lambda data: apply_mice(data, estimator=Lasso()),
        "MICE_ElasticNet": lambda data: apply_mice(data, estimator=ElasticNet()),
        "MICE_ExtraTrees": lambda data: apply_mice(data, estimator=ExtraTreesRegressor())
    }   

    results = run_experiment(data, masking_ratios, imputation_methods, n_runs=7, results_path="experiments/results")
    logger.info("MICE experiments completed")

    plot_results(
        results=results,
        metric="mse",
        title="MICE Imputation Performance",
        save_path="experiments/results"
    )
    logger.info("Results plot saved")
