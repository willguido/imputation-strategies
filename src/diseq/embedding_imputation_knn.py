import os
import pandas as pd
import numpy as np
import logging
from sklearn.metrics.pairwise import cosine_similarity
import time
import psutil
import csv
from utils.imputation import apply_knn_imputer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("knn_embedding_log.log"),
        logging.StreamHandler()
    ])

logger = logging.getLogger()

def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / (1024 ** 2)

# Converting string representation of embeddings into numpy array is needed
def parse_embedding(emb_str):
    emb_str = emb_str.strip("[]")
    return np.fromstring(emb_str, sep=" ")  

# Masking function for the dataset with embeddings
# Give names of the label and embedding column to mask
def mask_data(data, embedding_col, label_col, ratio, random_state=42):
    """
    returns:
    - the df with masked columns
    - boolean mask of masked row indices
    """
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

# Find the closest disease based on cosine similarity.
# The embeddings to compare against, either within dataset or MONDO label embeddings.
def find_closest_disease(imputed_vector, df_embeddings, embedding_col, label_col):
    """
    returns:
    - the label of the closest disease
    """
    all_embeddings_matrix = np.vstack(df_embeddings[embedding_col].values)
    
    similarities = cosine_similarity([imputed_vector], all_embeddings_matrix)[0]
    
    # Get index of the most similar entry
    closest_index = np.argmax(similarities)
    
    # The closest match
    return df_embeddings.iloc[closest_index][label_col]


if __name__ == "__main__":
    start_time = time.time()
    memory_start = get_memory_usage()

    logger.info("Loading datasets")
    mondo_embeddings_path = "" # Add label_embeddings.tsv file
    df_mondo_embeddings = pd.read_csv(mondo_embeddings_path, sep="\t")
    df_mondo_embeddings["embedding"] = df_mondo_embeddings["embedding"].apply(parse_embedding)

    logger.info(f"Loaded MONDO embeddings with shape: {df_mondo_embeddings.shape}")

    data_path = "" # Path to diag_subset with embeddings
    df = pd.read_csv(data_path)
    df['Label_Embedding'] = df['Label_Embedding'].apply(parse_embedding)
    logger.info(f"Loaded data with shape: {df.shape}")

    df_original = df.copy() #store original values before masking

    logger.info("Starting label and embedding masking")
    # Mask labels and embeddings
    mask_ratio = 0.1

    mask_start = time.time()
    df, mask_indices = mask_data(df, embedding_col='Label_Embedding', label_col='Label', ratio=mask_ratio)
    mask_end = time.time()
    logger.info(f"Masking completed in {mask_end - mask_start:.2f} seconds")

    # Save the masked data for future needs
    masked_data_path = "masked_data_diseq.csv"
    df[['Label', 'Label_Embedding']].to_csv(
        masked_data_path, index=False, quoting=csv.QUOTE_NONNUMERIC
    )
    logger.info(f"Masked data saved to {masked_data_path}")

    mask_indices_path = "mask_indices_diseq.csv"
    pd.DataFrame({'Mask_Indices': mask_indices}).to_csv(mask_indices_path, index=False)
    print(f"Mask indices saved to {mask_indices_path}")

    logger.info("Starting KNN imputation")

    # Flatten the embedding column into separate columns
    embeddings_expanded = pd.DataFrame(df['Label_Embedding'].tolist(), index=df.index)

    # Drop the original embedding column and add the columns from the previous step
    df_expanded = pd.concat([df.drop(columns=['SCIDPSEUDONYM', 'Label', 'Label_Embedding']), embeddings_expanded], axis=1)

    logger.info(f"Columns used for imputation: {df_expanded.columns.tolist()}")

    # Convert to numpy for imputation
    imputation_matrix = df_expanded.to_numpy()

    impute_start = time.time()
    imputed_data = apply_knn_imputer(imputation_matrix, n_neighbors=5)
    impute_end = time.time()
    logger.info(f"KNN imputation completed in {impute_end - impute_start:.2f} seconds")

    df_imputed = pd.DataFrame(imputed_data, columns=df_expanded.columns)

    # Number of features in the embeddings
    num_embedding_features = embeddings_expanded.shape[1]

    # Extract the imputed embedding columns and reshape them back into lists
    df['Imputed_Embedding'] = df_imputed.iloc[:, -num_embedding_features:].apply(lambda row: row.to_numpy(), axis=1)

    # Add other imputed values back to the original dataframe (excluding embeddings)
    df.update(df_imputed.iloc[:, :-num_embedding_features])

    imputed_data_path = "imputed_data_diseq.csv"
    df.to_csv(imputed_data_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
    logger.info(f"Imputed data saved to {imputed_data_path}")

    # Extract only masked rows for final evaluation
    df_masked_only = df.loc[mask_indices].copy()
    df_masked_only['Original_Masked_Label'] = df_original.loc[mask_indices, 'Label']

    logger.info("Finding closest diseases using non-masked data only")

    match_start = time.time()
    # Find closest disease within the non-masked subset
    df_masked_only['Closest_Disease_From_Labels'] = df_masked_only['Imputed_Embedding'].apply(
        lambda x: find_closest_disease(x, df_original.loc[~mask_indices], 'Label_Embedding', 'Label')
    )

    # Find closest disease using MONDO embeddings
    df_masked_only['Closest_Disease_From_Mondo'] = df_masked_only['Imputed_Embedding'].apply(
        lambda x: find_closest_disease(x, df_mondo_embeddings, 'embedding', 'label')
    )
    match_end = time.time()
    logger.info(f"Closest disease matching completed in {match_end - match_start:.2f} seconds")

    # Save imputed and true embedding for comparison later
    df_masked_only['Imputed_Embedding'] = df_masked_only['Imputed_Embedding'].apply(lambda x: x.tolist())

    df_masked_only['Original_Embedding'] = df_original.loc[df_masked_only.index, 'Label_Embedding'].apply(lambda x: x.tolist())

    output_path = "imputed_vs_true_diagnosis_masked_only_embedding.csv"
    df_masked_only[['Original_Masked_Label', 'Closest_Disease_From_Labels', 'Closest_Disease_From_Mondo', 'Original_Embedding', 'Imputed_Embedding']].to_csv(
    output_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
    logger.info(f"Results with embeddings exported to {output_path}")
    
    end_time = time.time()
    memory_end = get_memory_usage()
    total_runtime = end_time - start_time
    total_memory = memory_end - memory_start
    
    logger.info(f"Total script runtime: {total_runtime:.2f} seconds")
    logger.info(f"Total memory used: {total_memory:.2f} MB")
    
    comparison_file = "method_comparison.csv"
    with open(comparison_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Method", "Runtime (s)", "Memory (MB)"])
        writer.writerow(["KNN DISEQ Imputation", total_runtime, total_memory])
    
    logger.info(f"Comparison results saved to {comparison_file}")
