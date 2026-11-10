import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from data import load_triage_data


train_df, val_df, test_df = load_triage_data()
print(f"Data loaded! Total training rows: {len(train_df)}")

train_sample_df = train_df.sample(n=25000, random_state=42)

X_train_texts = train_sample_df["Issue"]
X_val_texts = val_df["Issue"]
X_test_texts = test_df["Issue"]

# 2. Initialize the Sentence Transformer model
print("\nLoading sentence-transformer model ('all-MiniLM-L6-v2')...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# 3. Check for text truncation limit
max_seq_len = model.max_seq_length
print(f"Model maximum sequence length: {max_seq_len} tokens")

# Quick check: how many training complaints exceed this limit? (using word count as a proxy)
word_counts = X_train_texts.apply(lambda x: len(str(x).split()))
exceeding_count = (word_counts > max_seq_len).sum()
print(f"Complaints exceeding length limit: {exceeding_count} out of {len(X_train_texts)} ({exceeding_count/len(X_train_texts)*100:.2f}%)")

# 4. Function to encode in batches and save to .npy
def encode_and_save(texts, filename, batch_size=256):
    print(f"\nEncoding {len(texts)} texts for {filename}...")
    
    embeddings = model.encode(texts.tolist(), batch_size=batch_size, show_progress_bar=True)
    
    
    np.save(filename, embeddings)
    print(f"Successfully saved {filename} with shape {embeddings.shape}!")
    return embeddings

# Run encoding for Train, Validation, and Test
train_embeddings = encode_and_save(X_train_texts, "train_embeddings.npy")
val_embeddings = encode_and_save(X_val_texts, "val_embeddings.npy")
test_embeddings = encode_and_save(X_test_texts, "test_embeddings.npy")

print("\n Step 1 Complete! All embeddings are safely stored locally as .npy files.")