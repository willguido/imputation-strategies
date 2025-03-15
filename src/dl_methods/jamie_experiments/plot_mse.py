import numpy as np
import matplotlib.pyplot as plt

# Paths for intermediate results
masking_ratios = [0.1, 0.3, 0.5, 0.7]
results_paths = [
    "results01/metrics_modality1.npy", 
    "results03/metrics_modality1.npy", 
    "results05/metrics_modality1.npy",
    "results07/metrics_modality1.npy"
]

results_paths_mod2 = [
    "results01/metrics_modality2.npy", 
    "results03/metrics_modality2.npy", 
    "results05/metrics_modality2.npy",
    "results07/metrics_modality2.npy"
]

mse_modality1 = []
mse_modality2 = []

# Load metrics and extract mse for modality 1
for path in results_paths:
    metrics = np.load(path, allow_pickle=True).item()
    mse_modality1.append(metrics['mse'])

# Load metrics and extract mse for modality 2
for path in results_paths_mod2:
    metrics = np.load(path, allow_pickle=True).item()
    mse_modality2.append(metrics['mse'])

plt.figure(figsize=(10, 6))
plt.plot(masking_ratios, mse_modality1, marker='o', label='Modality 1')
plt.plot(masking_ratios, mse_modality2, marker='o', label='Modality 2')

plt.title("JAMIE Imputation Performance")
plt.xlabel("Masking Ratio")
plt.ylabel("MSE")
plt.legend()
plt.grid(True)
plt.savefig("JAMIE_Imputation_Performance.png")
plt.close()