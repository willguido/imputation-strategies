import pandas as pd
from diseq import Diseq 

data_path = r""
output_path = r"" # Path for diag_subset with embeddings

t = Diseq()
t.download_data(data_dir=data_path)  
t.init_tool(data_dir=data_path, username="", password="") # Add username and password to use the tool

df = pd.read_csv("merged_data_v2_largest_subset_clean.csv")

# Select only relevant columns for embeddings
label_df = df[['SCIDPSEUDONYM', 'Label']].copy() 

# Apply embed_label to each label
label_df['Label_Embedding'] = label_df['Label'].apply(t.embed_label)

# Merge embeddings back into the original dataset
df = df.merge(label_df[['SCIDPSEUDONYM', 'Label_Embedding']], on='SCIDPSEUDONYM')

df.to_csv(output_path, index=False)

print(f"Data with embeddings added and saved to {output_path}")



