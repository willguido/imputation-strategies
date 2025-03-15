#!/usr/bin/env python

import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from utils.masking import mask_data

DATA_ORIGINAL = "balanced_complete_protein_subset_with_id.csv"
WORK_DIR = "" # path to protein subset directory
FIT_PY_PATH = "/AutoComplete/fit.py"
ID_COLUMN = "ID"

# Basic AutoComplete hyperparameters
BATCH_SIZE = 512
EPOCHS = 50
LR = 0.1
DEVICE = "cpu:0" 

# Load data and ignore ID column
df_original = pd.read_csv(DATA_ORIGINAL)
original_array = df_original.iloc[:, 1:].values

mask_ratios = np.arange(0.1, 1.0, 0.1)
mse_results = []

for ratio in mask_ratios:
    print(f"\nRunning ratio {ratio}")

    # Mask csv
    data_array = original_array.copy()
    masked_data, mask = mask_data(data_array, ratio=ratio, random_state=42)

    # Requirement of AC: Replace NaN with empty string
    df_masked = pd.DataFrame(masked_data, columns=df_original.columns[1:])
    df_masked = df_masked.replace({np.nan: ""})
    
    # Put the ID column back
    df_masked.insert(0, ID_COLUMN, df_original[ID_COLUMN])
    
    masked_csv_name = os.path.join(WORK_DIR, f"protein_subset_ratio_{ratio:.1f}_masked.csv")
    df_masked.to_csv(masked_csv_name, index=False)

    # Run AC
    imputed_csv_name = os.path.join(WORK_DIR, f"imputed_protein_subset_ratio_{ratio:.1f}.csv")

    cmd = [
        "python", FIT_PY_PATH,
        masked_csv_name,
        "--id_name", ID_COLUMN,
        "--batch_size", str(BATCH_SIZE),
        "--epochs", str(EPOCHS),
        "--lr", str(LR),
        "--device", DEVICE,
        "--save_imputed", 
        "--output", imputed_csv_name
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    
    # Calculate MSE
    df_imputed = pd.read_csv(imputed_csv_name)
    imputed_array = df_imputed.iloc[:, 1:].values.astype(float)

    masked_true = original_array[mask]
    masked_pred = imputed_array[mask]
    mse = mean_squared_error(masked_true, masked_pred)
    
    print(f"Mask ratio: {ratio} and MSE: {mse}")
    mse_results.append(mse)

# Plot the results
import matplotlib.pyplot as plt

plt.figure(figsize=(7,5))
plt.plot(mask_ratios, mse_results, marker='o')
plt.title("AutoComplete Imputation Performance")
plt.xlabel("Mask ratio")
plt.ylabel("MSE")
plt.grid(True)
plt.show()

plt.savefig(os.path.join(WORK_DIR, "AutoComplete_Imputation_Performance.png"))