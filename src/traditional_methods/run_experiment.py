import os
import json
import logging
from utils.masking import mask_data
from utils.evaluation import evaluate_imputation

logger = logging.getLogger(__name__)

def run_experiment(data, masking_ratios, imputation_methods, n_runs=5, results_path="results"):
    results_by_method = {}
    os.makedirs(results_path, exist_ok=True)

    for method_name, impute_func in imputation_methods.items():
        logger.info(f"Running experiments for method: {method_name}")
        method_results = []
        for ratio in masking_ratios:
            logger.info(f"  Masking ratio: {ratio}")
            for run in range(n_runs):
                random_state = run
                try:
                    # Mask the data
                    masked_data, mask = mask_data(data, ratio, random_state)
                    # Impute the data
                    imputed_data = impute_func(masked_data)
                    # Evaluate the results (MSE, MAE, RMSE, R^2)
                    metrics = evaluate_imputation(data.values, imputed_data, mask)

                    method_results.append({
                        'method': method_name,
                        'masking_ratio': ratio,
                        'run': run,
                        'metrics': metrics
                    })
                except Exception as e:
                    logger.error(f"Error during run {run + 1} for method {method_name}: {e}")
        
        # Save results by method
        method_results_file = os.path.join(results_path, f"{method_name.lower()}_results.json")
        with open(method_results_file, "w") as f:
            json.dump(method_results, f, indent=4)
        results_by_method[method_name] = method_results

    # Combine all results into a single list for plotting
    all_results = [item for sublist in results_by_method.values() for item in sublist]
    return all_results