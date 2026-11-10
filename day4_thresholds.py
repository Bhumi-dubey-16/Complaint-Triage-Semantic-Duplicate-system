import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

# 1. Load your hand-labeled candidate pairs
print("Loading hand-labeled pairs from CSV...")
labeled_df = pd.read_csv("candidate_pairs_to_label.csv")

# Safety check to ensure you've labeled the data
if "Is_Duplicate" not in labeled_df.columns:
    raise ValueError(
        " Error: 'Is_Duplicate' column not found! Please open 'candidate_pairs_to_label.csv', "
        "add an 'Is_Duplicate' column with 1s and 0s, and re-save the file before running this script."
    )

y_true = labeled_df["Is_Duplicate"].values
similarities = labeled_df["Similarity"].values

print(f"Total labeled pairs loaded: {len(labeled_df)}")
print(f"Total actual duplicates marked (1s): {sum(y_true)}")

# 2. Split into Tuning Set (First Half) and Held-Out Test Set (Second Half)
# This prevents data leakage so we honestly test our chosen threshold
split_idx = len(labeled_df) // 2
y_tune, sim_tune = y_true[:split_idx], similarities[:split_idx]
y_test, sim_test = y_true[split_idx:], similarities[split_idx:]

print(f"Tuning set size: {len(y_tune)} pairs")
print(f"Held-out test set size: {len(y_test)} pairs")

# 3. Sweep thresholds on the tuning set to find the optimal operating point
thresholds = np.linspace(0.70, 0.95, 26)
best_threshold = 0.80
best_f1 = 0

print("\n--- Running Threshold Tuning Sweep ---")
print(f"{'Threshold':<12} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
print("-" * 52)

for thresh in thresholds:
    # Convert continuous similarity scores into binary decisions (1 or 0) based on threshold
    preds = (sim_tune >= thresh).astype(int)
    
    p = precision_score(y_tune, preds, zero_division=0)
    r = recall_score(y_tune, preds, zero_division=0)
    f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
    
    print(f"{thresh:<12.2f} | {p:<10.2f} | {r:<10.2f} | {f1:<10.2f}")
    
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = thresh

print("-" * 52)
print(f" Optimal Threshold found on tuning set: {best_threshold:.2f} (F1: {best_f1:.4f})")

# 4. Final Evaluation on the Held-Out Test Set (The Unseen Exam)
test_preds = (sim_test >= best_threshold).astype(int)
final_p = precision_score(y_test, test_preds, zero_division=0)
final_r = recall_score(y_test, test_preds, zero_division=0)
final_f1 = 2 * (final_p * final_r) / (final_p + final_r) if (final_p + final_r) > 0 else 0

print("\n====================================================")
print(f"      HELD-OUT TEST EVALUATION (Threshold = {best_threshold:.2f})")
print("====================================================")
print(f"Final Test Precision : {final_p:.4f}")
print(f"Final Test Recall    : {final_r:.4f}")
print(f"Final Test F1-Score  : {final_f1:.4f}")
print("====================================================")