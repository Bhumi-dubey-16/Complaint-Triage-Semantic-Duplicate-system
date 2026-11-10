import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

class ComplaintTriagePipeline:
    def __init__(self, threshold=0.87):
        print("Initializing Sentence Transformer model...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.threshold = threshold
        self.cache_texts = []
        self.cache_embeddings = np.empty((0, 384), dtype='float32')

    def normalize(self, vectors):
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / norms

    def process_incoming_complaint(self, new_complaint_text):
        # 1. Generate embedding for the incoming complaint
        new_emb = self.model.encode([new_complaint_text]).astype('float32')
        new_emb_norm = self.normalize(new_emb) # Shape: (1, 384)

        # 2. Check semantic cache if items exist
        if len(self.cache_embeddings) > 0:
            # Multiply cache (N, 384) by incoming vector transposed (384, 1) -> Result: (N, 1)
            similarities = np.dot(self.cache_embeddings, new_emb_norm.T).flatten()
            best_match_idx = np.argmax(similarities)
            best_score = similarities[best_match_idx]

            if best_score >= self.threshold:
                return {
                    "Status": "Duplicate Found (Cache Hit)",
                    "SimilarityScore": float(best_score),
                    "MatchedExistingComplaint": self.cache_texts[best_match_idx],
                    "Action": "Merge with existing ticket"
                }

        # 3. Cache miss: add to cache for future lookups
        self.cache_texts.append(new_complaint_text)
        self.cache_embeddings = np.vstack([self.cache_embeddings, new_emb_norm])

        return {
            "Status": "Unique Complaint (Cache Miss)",
            "SimilarityScore": 0.0,
            "MatchedExistingComplaint": None,
            "Action": "Create new triage ticket"
        }
# Test the pipeline workflow

if __name__ == "__main__":
    pipeline = ComplaintTriagePipeline(threshold=0.70)

    c1 = "Street light is not working on 4th cross street."
    c2 = "Street light is not working on 4th cross."
    c3 = "Water leakage reported near the central station."

    print("\n--- Processing Complaint 1 ---")
    print(pipeline.process_incoming_complaint(c1))

    print("\n--- Processing Complaint 2 (Semantic Duplicate of C1) ---")
    print(pipeline.process_incoming_complaint(c2))

    print("\n--- Processing Complaint 3 (Distinct Issue) ---")
    print(pipeline.process_incoming_complaint(c3))