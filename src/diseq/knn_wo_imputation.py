import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import logging
import time
import psutil
import csv
from collections import Counter

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s', 
    handlers=[
        logging.FileHandler("knn_prediction_log.log"),
        logging.StreamHandler()
    ])
logger = logging.getLogger()

def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / (1024 ** 2)

# Masking function for the dataset with embeddings
def mask_data(data, embedding_col, label_col, ratio, random_state=42):
    np.random.seed(random_state)
    mask = np.random.rand(len(data)) < ratio
    
    # Mask "Label_Embedding"
    embedding_length = len(data[embedding_col].iloc[0])
    data.loc[mask, embedding_col] = data.loc[mask, embedding_col].apply(
        lambda x: np.full(embedding_length, np.nan)
    )
    
    # Mask "Label"
    data.loc[mask, label_col] = np.nan
    
    return data, mask

# Return most common label or None
def majority_vote(neighbors):
    neighbors = [label for label in neighbors if pd.notna(label)]
    counter = Counter(neighbors)
    return counter.most_common(1)[0][0] if counter else None

# Converting string representation of embeddings into numpy array is needed
def parse_embedding(emb_str):
    emb_str = emb_str.strip("[]")
    return np.fromstring(emb_str, sep=" ") 

if __name__ == "__main__":

    data_path = "" # Path to diag_subset with embeddings
    df = pd.read_csv(data_path)

    df['Label_Embedding'] = df['Label_Embedding'].apply(parse_embedding)

    protein_cols = [col for col in df.columns if col.startswith("Protein_")]
    df_proteins = df[protein_cols].to_numpy()
    embeddings = np.vstack(df['Label_Embedding'].values)

    df_original = df.copy()
    logger.info("Starting label and embedding masking")
    mask_ratio = 0.1  

    df, mask_indices = mask_data(df, embedding_col='Label_Embedding', label_col='Label', ratio=mask_ratio)
    logger.info("Masking completed")

    # Save masked data and indices
    masked_data_path = "no_imputation_masked_data.csv"
    df[['Label', 'Label_Embedding']].to_csv(masked_data_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
    logger.info(f"Masked data saved to {masked_data_path}")

    masked_rows = np.where(mask_indices)[0]
    mask_indices_path = "no_imputation_mask_indices.csv"
    pd.DataFrame({'Masked_Rows': masked_rows}).to_csv(mask_indices_path, index=False)

    # Compute nearest neighbors separately for embeddings and protein values
    start_time = time.time()
    memory_start = get_memory_usage()

    # KNN on embeddings with cosine distance
    logger.info("Computing nearest neighbors using cosine distance on embeddings")
    knn_embeddings = NearestNeighbors(n_neighbors=5, metric='cosine')
    knn_embeddings.fit(embeddings)
    neighbors_emb, indices_emb = knn_embeddings.kneighbors(embeddings)
    end_time_emb = time.time()
    memory_end_emb = get_memory_usage()
    runtime_emb = end_time_emb - start_time
    memory_usage_emb = memory_end_emb - memory_start
    logger.info(f"Embedding KNN completed in {runtime_emb:.2f} seconds, memory used: {memory_usage_emb:.2f} MB")

    # KNN on proteins with euclidean distance
    logger.info("Computing nearest neighbors using Euclidean distance on protein features")
    knn_proteins = NearestNeighbors(n_neighbors=5, metric='euclidean')
    knn_proteins.fit(df_proteins)
    neighbors_prot, indices_prot = knn_proteins.kneighbors(df_proteins)
    end_time_prot = time.time()
    memory_end_prot = get_memory_usage()
    runtime_prot = end_time_prot - end_time_emb
    memory_usage_prot = memory_end_prot - memory_end_emb
    logger.info(f"Protein KNN completed in {runtime_prot:.2f} seconds, memory used: {memory_usage_prot:.2f} MB")

    # Store results
    results = []
    for i in range(len(df)):
        label = df.iloc[i]['Label']
        
        # Get nearest labels
        nn_labels_emb = df.iloc[indices_emb[i]]['Label'].dropna().tolist()
        nn_labels_prot = df.iloc[indices_prot[i]]['Label'].dropna().tolist()
        
        if pd.isna(label):
            # Predict and get the most common label
            predicted_label_emb = majority_vote(nn_labels_emb)
            predicted_label_prot = majority_vote(nn_labels_prot)
        else:
            # If not masked, keep the original label
            predicted_label_emb = label
            predicted_label_prot = label
        
        results.append({
            'Row_Index': i,
            'True_Label': df_original.iloc[i]['Label'],
            'Predicted_Label_Embedding': predicted_label_emb,
            'Predicted_Label_Protein': predicted_label_prot
        })

    # Save results
    results_df = pd.DataFrame(results)

    # Keep only masked rows
    masked_indices_set = set(masked_rows)
    results_df = results_df[results_df["Row_Index"].isin(masked_indices_set)]

    results_path = "no_imputation_predicted_vs_true_labels.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"Filtered predictions saved to {results_path}")

    # Log computational resources
    comparison_file = "method_comparison.csv"
    with open(comparison_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Method", "Runtime (s)", "Memory (MB)"])
        writer.writerow(["Cosine KNN (Embeddings)", runtime_emb, memory_usage_emb])
        writer.writerow(["Euclidean KNN (Proteins)", runtime_prot, memory_usage_prot])
    logger.info(f"Comparison results saved to {comparison_file}")
