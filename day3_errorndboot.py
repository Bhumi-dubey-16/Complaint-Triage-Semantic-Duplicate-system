import numpy as np
import pandas as pd
from data import load_triage_data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 1. Load validation data and text
_, val_df, _ = load_triage_data()
X_val_texts = val_df["Issue"]
y_val = val_df["Product"]

# 2. Get TF-IDF predictions (re-running TF-IDF vectorizer fit on train, transform on val for fair comparison)
train_df, _, _ = load_triage_data()
train_sample_df = train_df.sample(n=25000, random_state=42)

vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=5, sublinear_tf=True)
X_train_vec = vectorizer.fit_transform(train_sample_df["Issue"])
X_val_vec = vectorizer.transform(X_val_texts)

tfidf_clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=10.0)
tfidf_clf.fit(X_train_vec, train_sample_df["Product"])
y_val_pred_tfidf = tfidf_clf.predict(X_val_vec)

# 3. Get Embedding predictions (loaded from disk)
X_train_emb = np.load("train_embeddings.npy")
X_val_emb = np.load("val_embeddings.npy")

emb_clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=10.0)
emb_clf.fit(X_train_emb, train_sample_df["Product"])
y_val_pred_emb = emb_clf.predict(X_val_emb)

# 4. Find Disagreements (Where exactly one model is right and the other is wrong)
y_val_array = y_val.values

tfidf_correct = (y_val_pred_tfidf == y_val_array)
emb_correct = (y_val_pred_emb == y_val_array)

# TF-IDF right, Embeddings wrong
tfidf_wins_mask = tfidf_correct & (~emb_correct)
# Embeddings right, TF-IDF wrong
emb_wins_mask = (~tfidf_correct) & emb_correct

# Build a DataFrame of disagreements for forensic review
disagreements_df = pd.DataFrame({
    "Complaint_Text": X_val_texts,
    "True_Label": y_val_array,
    "TFIDF_Prediction": y_val_pred_tfidf,
    "Embedding_Prediction": y_val_pred_emb
})

# Filter to rows where they disagreed and one of them was correct
conflict_df = disagreements_df[tfidf_wins_mask | emb_wins_mask].copy()
conflict_df["Winner"] = np.where(tfidf_wins_mask[conflict_df.index], "TF-IDF Won", "Embeddings Won")

# Save to CSV so you can inspect them easily
conflict_df.to_csv("model_disagreements.csv", index=False)
print(f"🔍 Found {len(conflict_df)} disagreement cases where one model was right and the other wrong.")
print("Saved details to 'model_disagreements.csv' for review!")