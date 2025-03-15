import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import umap
from jamie import JAMIE
from utils.masking import mask_data

# Load original data and split into two modalities
data = pd.read_csv("largest_balanced_complete_protein_subset.csv").values
data1 = data[:, :105]
data2 = data[:, 105:]

# Scale the full data before masking
scaler1 = StandardScaler()
scaler2 = StandardScaler()
data1_scaled = scaler1.fit_transform(data1)
data2_scaled = scaler2.fit_transform(data2)

# Masking on the scaled data and replace NaNs with 0 (JAMIE requirement)
masked_data1, mask1 = mask_data(data1_scaled, ratio=0.5)
masked_data2, mask2 = mask_data(data2_scaled, ratio=0.5)
masked_data1 = np.nan_to_num(masked_data1, nan=0.0)
masked_data2 = np.nan_to_num(masked_data2, nan=0.0)

# Save the masked and scaled data
np.save("/results05/data1_scaled.npy", masked_data1)
np.save("/results05/data2_scaled.npy", masked_data2)
np.save("/results05/mask1.npy", mask1)
np.save("/results05/mask2.npy", mask2)
print("Masked and scaled data and masks saved")

# Run JAMIE integration and imputation
jm = JAMIE(min_epochs=500)
integrated_data = jm.fit_transform(dataset=[masked_data1, masked_data2])
np.save("/results05/integrated_data.npy", integrated_data)
print("Integrated data saved")

data1_imputed_scaled = jm.modal_predict(masked_data2, 1)
data2_imputed_scaled = jm.modal_predict(masked_data1, 0)

# Inverse transform to original scale for evaluation
data1_imputed = scaler1.inverse_transform(data1_imputed_scaled)
data2_imputed = scaler2.inverse_transform(data2_imputed_scaled)
np.save("/results05/data1_imputed.npy", data1_imputed)
print("data1_imputed saved")
np.save("/results05/data2_imputed.npy", data2_imputed)
print("data2_imputed saved")

# Evaluate on missing entries only
mse1 = mean_squared_error(data1[mask1], data1_imputed[mask1])
mse2 = mean_squared_error(data2[mask2], data2_imputed[mask2])
mae1 = mean_absolute_error(data1[mask1], data1_imputed[mask1])
mae2 = mean_absolute_error(data2[mask2], data2_imputed[mask2])
rmse1 = np.sqrt(mse1)
rmse2 = np.sqrt(mse2)
r2_1 = r2_score(data1[mask1], data1_imputed[mask1])
r2_2 = r2_score(data2[mask2], data2_imputed[mask2])

print(f"MSE Modality 1: {mse1}, MSE Modality 2: {mse2}")
print(f"MAE Modality 1: {mae1}, MAE Modality 2: {mae2}")
print(f"RMSE Modality 1: {rmse1}, RMSE Modality 2: {rmse2}")
print(f"R^2 Modality 1: {r2_1}, R^2 Modality 2: {r2_2}")

metrics1 = {'mse': mse1, 'mae': mae1, 'rmse': rmse1, 'r2': r2_1}
metrics2 = {'mse': mse2, 'mae': mae2, 'rmse': rmse2, 'r2': r2_2}
np.save("/results05/metrics_modality1.npy", metrics1)
print("Metrics for modality 1 saved")
np.save("/results05/metrics_modality2.npy", metrics2)
print("Metrics for modality 2 saved")

# Visualization using UMAP
umap_model = umap.UMAP(n_components=2, random_state=42)
data1_umap = umap_model.fit_transform(masked_data1)
np.save("/results05/data1_umap.npy", data1_umap)
print("data1_umap saved")
data2_umap = umap_model.fit_transform(masked_data2)
np.save("/results05/data2_umap.npy", data2_umap)
print("data2_umap saved")
data_integrated_umap = umap_model.fit_transform(integrated_data[0])
np.save("/results05/data_integrated_umap.npy", data_integrated_umap)
print("data_integrated_umap saved")

# Plot measured vs imputed for modality 1
combined_mod1 = np.vstack([masked_data1, data1_imputed_scaled])
umap_model_combined = umap.UMAP(n_components=2, random_state=42)
combined_mod1_umap = umap_model_combined.fit_transform(combined_mod1)
n_measured = masked_data1.shape[0]
data1_measured_umap = combined_mod1_umap[:n_measured, :]
data1_imputed_umap = combined_mod1_umap[n_measured:, :]

plt.figure(figsize=(10, 5))
plt.scatter(data1_measured_umap[:, 0], data1_measured_umap[:, 1], label='Measured Modality 1', alpha=0.6)
plt.scatter(data1_imputed_umap[:, 0], data1_imputed_umap[:, 1], label='Imputed Modality 1', alpha=0.6)
plt.title("Measured vs Imputed Modalities")
plt.legend()
plt.savefig("/results05/UMAP_Imputed_JAMIE.png")
print("Plot UMAP_Imputed_JAMIE.png saved")
plt.close()
