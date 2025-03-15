import pandas as pd
from utils import load_disease_ontology, build_ontology_graph, convert_to_undirected, compute_doid_distance, get_missing_reason

# Load DO and build the undirected graph
ontology = load_disease_ontology()
graph_directed = build_ontology_graph(ontology)
graph = convert_to_undirected(graph_directed)

print(f"Ontology Graph loaded with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")

file_path = "" # Path to the file with manually filled in DOIDs
df = pd.read_csv(file_path, delimiter=";")

# Ensure DOID columns are properly formatted
df['DOID_original'] = df['DOID_original'].astype(str).str.strip().replace("nan", "")
df['DOID_imputed_labels'] = df['DOID_imputed_labels'].astype(str).str.strip().replace("nan", "")
df['DOID_imputed_mondo'] = df['DOID_imputed_mondo'].astype(str).str.strip().replace("nan", "")

# Compute DO distances
df['DO_Distance_Labels'] = df.apply(
    lambda row: compute_doid_distance(graph, row['DOID_original'], row['DOID_imputed_labels']) 
    if pd.notna(row['DOID_original']) and pd.notna(row['DOID_imputed_labels']) else None, 
    axis=1
)

df['DO_Distance_Mondo'] = df.apply(
    lambda row: compute_doid_distance(graph, row['DOID_original'], row['DOID_imputed_mondo']) 
    if pd.notna(row['DOID_original']) and pd.notna(row['DOID_imputed_mondo']) else None, 
    axis=1
)

# Add missing reason if distance is None
df['Missing_Reason_Labels'] = df.apply(
    lambda row: get_missing_reason(graph, row['DOID_original'], row['DOID_imputed_labels'])
    if pd.isna(row['DO_Distance_Labels']) else None, axis=1
)

df['Missing_Reason_Mondo'] = df.apply(
    lambda row: get_missing_reason(graph, row['DOID_original'], row['DOID_imputed_mondo'])
    if pd.isna(row['DO_Distance_Mondo']) else None, axis=1
)

output_path = "" # Path to save the resulting distances
df.to_csv(output_path, index=False, sep=";")

print(f"DO distances computed and saved to '{output_path}'")