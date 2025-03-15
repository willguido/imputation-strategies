import pandas as pd
import numpy as np

# Group columns and rows into ranges of missingness
# Analyze subsets systematically to create a fully complete data first
# Algorithm for finding the largest complete (no missing values) and balanced (row*column score) subset
def find_largest_balanced_subset(filepath, output_path, column_prefix):
    data = pd.read_csv(filepath)
    
    # Filter for specific columns using the prefix
    filtered_columns = [col for col in data.columns if col.startswith(column_prefix)]
    filtered_data = data[filtered_columns]
    print(f"Filtered data shape: {filtered_data.shape}")

    # 1) Flatten missing data by reordering rows and columns
    # Sort columns by missingness, ascending
    columns_sorted_by_missingness = filtered_data.isnull().mean().sort_values().index
    print(f"Sorted columns by missingness: {len(columns_sorted_by_missingness)} columns")
    data_sorted_columns = filtered_data[columns_sorted_by_missingness]
    print(f"Data shape after sorting columns: {data_sorted_columns.shape}")

    # Sort rows by missingness, ascending
    rows_sorted_by_missingness = data_sorted_columns.isnull().mean(axis=1).sort_values().index
    print(f"Sorted rows by missingness: {len(rows_sorted_by_missingness)} rows")
    data_sorted = data_sorted_columns.loc[rows_sorted_by_missingness]
    print(f"Data shape after sorting rows: {data_sorted.shape}")

    # 2) Identify a balanced complete subset
    # Start with all rows and progressively reduce columns
    largest_subset = None
    max_score = 0  # Score based on rows * columns

    for i in range(len(data_sorted.columns), 0, -1):
        subset = data_sorted.iloc[:, :i].dropna()
        rows, cols = subset.shape
        score = rows * cols
        print(f"Checking first {i} columns: subset shape: {subset.shape}, score: {score}")
        if score > max_score:
            largest_subset = subset
            max_score = score
            print(f"New largest balanced subset: {largest_subset.shape}")

    # 3) Largest balanced complete subset found
    largest_subset.to_csv(output_path, index=False)

    print(f"Original data shape: {filtered_data.shape}")
    print(f"Largest balanced complete subset shape: {largest_subset.shape}")

    return largest_subset

# Specify number of rows and columns, and randomly sample
def random_sampling(data, n_rows, n_cols, random_state=42):
    np.random.seed(random_state)

    row_indices = np.random.choice(data.index, size=min(n_rows, len(data)), replace=False)

    col_indices = np.random.choice(data.columns, size=min(n_cols, len(data.columns)), replace=False)

    subset = data.loc[row_indices, col_indices]
    return subset