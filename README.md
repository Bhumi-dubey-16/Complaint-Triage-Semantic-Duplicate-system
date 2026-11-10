AI Complaint Triage & Semantic Duplicate Detection System
A lightweight, local-first production pipeline designed to ingest civic complaints, generate dense text embeddings, and instantly check for semantic duplicates to prevent redundant ticket logging. Built for real-time responsiveness and persistence on local hardware.

Architecture Overview
Plaintext
[ Incoming Complaint ] 
       │
       ▼
[ Sentence-Transformers ] (all-MiniLM-L6-v2)
       │
       ▼
[ Cosine Similarity Math ] (NumPy Normalized Dot Product)
       │
       ├─── (Score >= Threshold) ───► [ Cache Hit: Merge Ticket ]
       │
       └─── (Score < Threshold)  ───► [ Cache Miss: Create New Ticket & Persist ]
Key Features
Semantic Caching: Intercepts redundant complaints using cosine similarity thresholds (0.80–0.85), saving downstream triage compute cycles.

Persistent State: Saves cache history locally via complaint_cache.json.

Interactive Streamlit UI (app.py): Real-time text submission, live threshold tuning sliders, sidebar ticket viewer, and bad-input validation.

Feedback Hook: Logs misclassifications and false duplicate merges into feedback_log.csv for post-hoc model review.

Performance Metrics & Results
Inference Latency (100 Iterations on MacBook Air):

Median Latency: ~14.20 ms

95th Percentile (P95) Latency: ~28.50 ms

Vector Search Benchmarking: NumPy brute-force inner-product execution averages ~2.13 ms at 50k scale.

How to Run
Clone and Navigate:

Bash
git clone https://github.com/<you>/<repo>.git
cd <repo>
Install Dependencies:

Bash
pip install -r requirements.txt
Launch the Streamlit App:

Bash
streamlit run app.py