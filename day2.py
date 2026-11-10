import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data import load_triage_data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, confusion_matrix

# 1. Load data
train_df, val_df, test_df = load_triage_data()
print(f"Data loaded successfully! Total training rows available: {len(train_df)}")

# --- SUB-SAMPLE FOR FAST TRAINING ---
train_sample_df = train_df.sample(n=25000, random_state=42)

# 2. Separate Features (X) and Targets (y)
X_train = train_sample_df["Issue"]
y_train = train_sample_df["Product"]

X_val = val_df["Issue"]
y_val = val_df["Product"]

X_test = test_df["Issue"]
y_test = test_df["Product"]

# 3. Vectorization with tuned settings
print("\nStarting vectorization...")
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    min_df=5,
    sublinear_tf=True
)

X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)
X_test_vec = vectorizer.transform(X_test)

# 4. Tune Regularization Parameter (C) on Validation Set
print("\n--- Tuning Regularization Parameter (C) ---")
c_values = [0.1, 1.0, 10.0]
best_c = 1.0
best_val_f1 = 0.0

for c in c_values:
    print(f"Testing C = {c}...")
    temp_clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=c)
    temp_clf.fit(X_train_vec, y_train)
    
    val_preds = temp_clf.predict(X_val_vec)
    score = f1_score(val_preds, y_val, average="macro", zero_division=0)
    print(f"  -> Macro-F1 with C={c}: {score:.4f}")
    
    if score > best_val_f1:
        best_val_f1 = score
        best_c = c

print(f"\nBest C chosen: {best_c} with Validation Macro-F1: {best_val_f1:.4f}")

# 5. Train Final Model with Best C
print(f"\n--- Training Final Logistic Regression Model (C={best_c}) ---")
final_clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=best_c)
final_clf.fit(X_train_vec, y_train)

# 6. Final Evaluation on the Untouched Test Set
print("\n--- Final Evaluation on Test Set ---")
y_test_pred = final_clf.predict(X_test_vec)
test_macro_f1 = f1_score(y_test_pred, y_test, average="macro", zero_division=0)
test_weighted_f1 = f1_score(y_test_pred, y_test, average="weighted", zero_division=0)

print(f"Final Test Macro-F1 Score: {test_macro_f1:.4f}")
print(f"Final Test Weighted-F1 Score: {test_weighted_f1:.4f}")

# 7. Save Results to CSV
results_data = {
    "Model": ["Logistic Regression (Tuned)"],
    "Best_C": [best_c],
    "Test_Macro_F1": [test_macro_f1],
    "Test_Weighted_F1": [test_weighted_f1]
}
results_df = pd.DataFrame(results_data)
results_df.to_csv("day2_results.csv", index=False)
print("\nResults successfully saved to 'day2_results.csv'!")