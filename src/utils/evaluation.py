from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os
import pandas as pd

# Extract the masked values for comparison
# return original and imputed values for the masked positions
def extract_masked_values(original_data, imputed_data, mask):
    original_values = original_data[mask]
    imputed_values = imputed_data[mask]
    return original_values, imputed_values

# Evaluation metrics only on the masked entries
def calculate_mse(original_data, imputed_data, mask):
    original_values, imputed_values = extract_masked_values(original_data, imputed_data, mask)
    return mean_squared_error(original_values, imputed_values)

def calculate_rmse(original_values, imputed_values, mask):
    return root_mean_squared_error(original_values, imputed_values)

def evaluate_imputation(original_data, imputed_data, mask):
    original_values, imputed_values = extract_masked_values(original_data, imputed_data, mask)

    mse = mean_squared_error(original_values, imputed_values)
    mae = mean_absolute_error(original_values, imputed_values)
    rmse = root_mean_squared_error(original_values, imputed_values)
    r2 = r2_score(original_values, imputed_values)
    return {'mse': mse, 'mae': mae, 'rmse': rmse, 'r2': r2}


def plot_results(results, metric="mse", title="Imputation Performance", save_path="results"):
    df = pd.DataFrame(results)

    # Ensure the save directory exists
    os.makedirs(save_path, exist_ok=True)

    if 'method' not in df.columns:
        raise KeyError("The 'method' column is missing from the results")

    plt.figure(figsize=(12, 8))

    # Group by method and masking ratio
    for method, group in df.groupby('method'):
        metric_values = group.groupby('masking_ratio')['metrics'].apply(
            lambda x: sum([m[metric] for m in x]) / len(x)
        )
        plt.plot(metric_values.index, metric_values.values, marker='o', label=method)

    plt.title(title)
    plt.xlabel("Masking Ratio")
    plt.ylabel(metric.upper())
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.grid(True)

    plot_file = os.path.join(save_path, f"{title.replace(' ', '_')}.png")
    plt.savefig(plot_file, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to: {plot_file}")


def plot_mse_vs_mask_ratio(mask_ratios, mse_results, title="MSE vs Masking Ratio", best_estimator=None):
    plt.figure(figsize=(10, 6))
    plt.plot(mask_ratios, mse_results, marker='o', label=f'Best Estimator: {best_estimator}' if best_estimator else None)
    plt.title(title)
    plt.xlabel('Masking Ratio')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.grid()
    plt.legend()
    plt.show()