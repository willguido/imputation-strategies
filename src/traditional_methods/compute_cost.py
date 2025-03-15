#!/usr/bin/env python

import os
import time
import psutil
import csv
import logging
import numpy as np
import pandas as pd
import tracemalloc # for peak memory tracking

from utils.imputation import (
    apply_knn_imputer,
    apply_mice,
    apply_mean_imputation,
    apply_missforest
)
from utils.masking import mask_data


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("traditional_method_cost_comparison.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_memory_usage_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024**2)

# Give the method name and measure runtime, total memory change, and peak memory usage of it
def measure_method_cost(method_name: str, func, data, **kwargs):
    start_time = time.time()
    start_mem = get_memory_usage_mb()

    tracemalloc.start()

    output = func(data, **kwargs)

    end_time = time.time()
    end_mem = get_memory_usage_mb() 

    current_mem, peak_mem = tracemalloc.get_traced_memory()  
    tracemalloc.stop() 

    total_mem_used = round(end_mem - start_mem, 3)
    peak_mem_used = round(peak_mem / (1024**2), 3) 

    return {
        "method": method_name,
        "runtime_s": round(end_time - start_time, 3),
        "total_memory_mb": total_mem_used,  # psutil
        "peak_memory_mb": peak_mem_used,  # tracemalloc
    }

def main():
    data_path = "largest_balanced_complete_protein_subset.csv"
    if not os.path.exists(data_path):
        logger.error(f"Data file not found at {data_path}")
        return

    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Data shape: {df.shape}")

    logger.info("Masking data with ratio 0.1")
    df_masked, mask_array = mask_data(df, ratio=0.1, random_state=42)
    logger.info("Masking completed")

    masked_np = df_masked.to_numpy(dtype=float)

    # Run each method to measure the costs
    results = []

    # Mean imputation
    cost = measure_method_cost("MeanImputation", apply_mean_imputation, df_masked)
    results.append(cost)
    logger.info(f"{cost}")

    # KNN with 15 neighbors
    cost = measure_method_cost("KNN-15", apply_knn_imputer, masked_np, n_neighbors=15)
    results.append(cost)
    logger.info(f"{cost}")

    # MICE with ExtraTrees estimator
    from sklearn.ensemble import ExtraTreesRegressor
    cost = measure_method_cost("MICE-ExtraTrees", apply_mice, df_masked, estimator=ExtraTreesRegressor())
    results.append(cost)
    logger.info(f"{cost}")

    # MissForest with random forest regressor
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    cost = measure_method_cost(
        "MissForest-RF",
        apply_missforest,
        df_masked,
        classifier=RandomForestClassifier(),  
        regressor=RandomForestRegressor()) 
    results.append(cost)
    logger.info(f"{cost}")

    # Save results
    comparison_file = "computational_comparison.csv"
    with open(comparison_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "Runtime (s)", "Memory (MB)"])
        for r in results:
            writer.writerow([r["method"], r["runtime_s"], r["memory_mb"]])

    logger.info(f"Comparison results saved to {comparison_file}")
    logger.info("Costs computed successfully")

if __name__ == "__main__":
    main()
