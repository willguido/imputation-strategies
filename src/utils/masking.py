import numpy as np

# Mask the dataset at given ratio and random state for reproducibility 
def mask_data(data, ratio, random_state=42):
    np.random.seed(random_state)
    mask = np.random.rand(*data.shape) < ratio
    masked_data = data.copy()
    masked_data[mask] = np.nan
    return masked_data, mask