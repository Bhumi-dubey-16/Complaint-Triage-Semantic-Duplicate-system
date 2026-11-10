import numpy as np
import pandas as pd
from data import load_triage_data
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

# 1. Load data to get validation targets (y_val)
_, val_df, _ = load_triage_data()
y_val = val_df["Product"]


train_df, _, _ = load_triage_data()
train_sample_df = train_df.sample(n=25000, random_state=42)
y_train = train_sample_df["Product"]

# 2. Load the cached .npy embeddings from disk
print("Loading cached embedding vectors from disk...")
X_train_emb = np.load("train_embeddings.npy")
X_val_emb = np.load("val_embeddings.npy")
print(f"Train embeddings shape: {X_train_emb.shape}")
print(f"Validation embeddings shape: {X_val_emb.shape}")

# 3. Train Logistic Regression on Embeddings
print("\n--- Training Logistic Regression on Sentence Embeddings ---")
emb_clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=10.0)
emb_clf.fit(X_train_emb, y_train)

# 4. Evaluate on Validation Set
print("Evaluating on validation set...")
y_val_pred_emb = emb_clf.predict(X_val_emb)
emb_macro_f1 = f1_score(y_val_pred_emb, y_val, average="macro", zero_division=0)
emb_weighted_f1 = f1_score(y_val_pred_emb, y_val, average="weighted", zero_division=0)

print(f"Embedding Model Validation Macro-F1: {emb_macro_f1:.4f}")
print(f"Embedding Model Validation Weighted-F1: {emb_weighted_f1:.4f}")

# 5. Build Side-by-Side Comparison Table
tfidf_val_macro_f1 = 0.5811

comparison_data = {
    "Feature Representation": ["TF-IDF (Day 2)", "Sentence Embeddings (Day 3)"],
    "Vector Dimensions": ["~400 (Sparse)", "384 (Dense)"],
    "Validation Macro-F1": [tfidf_val_macro_f1, emb_macro_f1]
}

comparison_df = pd.DataFrame(comparison_data)
print("\n========================================")
print("       MODEL COMPARISON TABLE           ")
print("========================================")
print(comparison_df.to_string(index=False))
print("========================================")

# 6. Bootstrap Confidence Interval Function
def bootstrap_macro_f1(y_true, y_pred, n_iterations=1000, sample_fraction=1.0, random_state=42):
    np.random.seed(random_state)
    scores = []
    n_samples = int(len(y_true) * sample_fraction)
    
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    
    for _ in range(n_iterations):
        indices = np.random.choice(len(y_true_arr), size=n_samples, replace=True)
        sample_true = y_true_arr[indices]
        sample_pred = y_pred_arr[indices]
        
        score = f1_score(sample_true, sample_pred, average="macro", zero_division=0)
        scores.append(score)
        
    lower_bound = np.percentile(scores, 2.5)
    upper_bound = np.percentile(scores, 97.5)
    mean_score = np.mean(scores)
    
    return mean_score, lower_bound, upper_bound

print("\n--- Computing Bootstrap Confidence Interval (Embedding Model) ---")
mean_emb, ci_emb_low, ci_emb_high = bootstrap_macro_f1(y_val, y_val_pred_emb)
print(f"Embedding Model Macro-F1:  {mean_emb:.4f} (95% CI: [{ci_emb_low:.4f}, {ci_emb_high:.4f}])")