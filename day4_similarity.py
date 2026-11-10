import numpy as np
import pandas as pd
from data import load_triage_data

# 1. Load validation texts and embeddings
_, val_df, _ = load_triage_data()
X_val_texts = val_df["Issue"].reset_index(drop=True)
X_val_emb = np.load("val_embeddings.npy")

print(f"Total validation embeddings loaded: {X_val_emb.shape}")

# 2. Take a random sample of 50,000 rows from the validation set as our working pool
np.random.seed(42)
pool_indices = np.random.choice(len(X_val_emb), size=50000, replace=False)
X_pool_texts = X_val_texts.iloc[pool_indices].reset_index(drop=True)
X_pool_emb = X_val_emb[pool_indices]

print(f"Created working search pool of shape: {X_pool_emb.shape}")

# 3. Normalize embeddings so dot product = cosine similarity
print("Normalizing embedding vectors...")
norms = np.linalg.norm(X_pool_emb, axis=1, keepdims=True)
X_pool_normalized = X_pool_emb / norms

# 4. Take a sample of 1,000 query complaints from the pool
query_indices = np.random.choice(len(X_pool_normalized), size=1000, replace=False)
queries = X_pool_normalized[query_indices]

# Compute similarity matrix (1000 queries vs 50,000 pool vectors)
print("Computing similarity matrix using NumPy dot product...")
similarity_matrix = np.dot(queries, X_pool_normalized.T)

# 5. Extract candidate pairs and bucket them into similarity bands
print("Extracting candidate pairs across similarity bands...")
pairs = []

for i, query_idx in enumerate(query_indices):
    sims = similarity_matrix[i]
    # Sort descending, ignore the first one (it's the exact same document with score 1.0)
    sorted_neighbor_indices = np.argsort(sims)[::-1]
    
    for neighbor_idx in sorted_neighbor_indices[1:6]: # Look at top 5 neighbors per query
        score = sims[neighbor_idx]
        
        # Define similarity bands
        if score >= 0.95:
            band = "0.95+"
        elif score >= 0.90:
            band = "0.90-0.95"
        elif score >= 0.85:
            band = "0.85-0.90"
        elif score >= 0.80:
            band = "0.80-0.85"
        elif score >= 0.70:
            band = "0.70-0.80"
        else:
            continue  # Skip low scores
            
        pairs.append({
            "Query_Text": X_pool_texts[query_idx],
            "Neighbor_Text": X_pool_texts[neighbor_idx],
            "Similarity": score,
            "Band": band
        })

pairs_df = pd.DataFrame(pairs)

if len(pairs_df) == 0:
    raise ValueError(" No pairs found above 0.70 similarity! Try expanding the pool size.")

# Safely sample up to 40 pairs per band without breaking columns
sampled_list = []
for band_name, group in pairs_df.groupby("Band"):
    sample_size = min(len(group), 40)
    sampled_list.append(group.sample(n=sample_size, random_state=42))

sampled_pairs = pd.concat(sampled_list).reset_index(drop=True)
sampled_pairs.to_csv("candidate_pairs_to_label.csv", index=False)

print(f"\nSaved {len(sampled_pairs)} stratified candidate pairs to 'candidate_pairs_to_label.csv'!")
print(sampled_pairs["Band"].value_counts())