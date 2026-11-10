import os
import json
import pandas as pd
import streamlit as st
import torch
from sentence_transformers import SentenceTransformer, util

# File to persist complaint cache across runs
CACHE_FILE = "complaint_cache.json"
FEEDBACK_FILE = "feedback_log.csv"

# Page Configuration
st.set_page_config(
    page_title="AI Complaint Triage System",
    page_icon="🤖",
    layout="centered"
)

@st.cache_resource
def load_model():
    """Load the Sentence Transformer model once and cache it."""
    return SentenceTransformer('all-MiniLM-L6-v2')

def load_cache():
    """Load existing complaints from JSON file if it exists."""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    return []

def save_cache(cache_data):
    """Save complaint cache to JSON file for persistence."""
    with open(CACHE_FILE, "w") as f:
        json.dump(cache_data, f, indent=4)

def handle_bad_inputs(text):
    """Validate input text for empty strings or extreme lengths."""
    if not text or len(text.strip()) == 0:
        return "Error: Input cannot be empty. Please enter a valid complaint description."
    if len(text.split()) > 2000:
        return "Error: Input exceeds the maximum allowed length of 2,000 words."
    return None

# Initialize Model and Persistent Cache
model = load_model()
if "complaint_cache" not in st.session_state:
    st.session_state.complaint_cache = load_cache()

# UI Header
st.title("AI Complaint Triage & Duplicate Detection")
st.markdown("Submit a new civic complaint to check against existing tickets using semantic text embeddings.")

# Text Input Form
with st.form("triage_form"):
    new_complaint = st.text_area("Enter Complaint Description:", placeholder="e.g., Street light is completely dead on 4th cross street.")
    similarity_threshold = st.slider("Duplicate Similarity Threshold", min_value=0.50, max_value=0.99, value=0.85, step=0.01)
    submitted = st.form_submit_button("Analyze & Process Complaint")

if submitted:
    # 1. Run Bad Input Validation First
    error_msg = handle_bad_inputs(new_complaint)
    if error_msg:
        st.error(error_msg)
    else:
        with st.spinner("Analyzing semantics..."):
            # Generate embedding for the new complaint
            new_embedding = model.encode(new_complaint, convert_to_tensor=True)
            
            best_match = None
            highest_score = 0.0
            
            # Check against stored cache if it's not empty
            if st.session_state.complaint_cache:
                stored_texts = [item["text"] for item in st.session_state.complaint_cache]
                stored_embeddings = model.encode(stored_texts, convert_to_tensor=True)
                
                # Compute cosine similarities
                cos_scores = util.cos_sim(new_embedding, stored_embeddings)[0]
                best_match_idx = torch.argmax(cos_scores).item()
                highest_score = cos_scores[best_match_idx].item()
                
                if highest_score >= similarity_threshold:
                    best_match = st.session_state.complaint_cache[best_match_idx]["text"]

            # Evaluate Action Result
            if best_match:
                result = {
                    "Status": "Duplicate Found (Cache Hit)",
                    "SimilarityScore": float(highest_score),
                    "MatchedExistingComplaint": best_match,
                    "Action": "Merge with existing ticket"
                }
                st.error(f"**Status:** {result['Status']}")
                st.info(f"**Similarity Score:** `{result['SimilarityScore']:.4f}`")
                st.warning(f"**Matched Existing Ticket:** {result['MatchedExistingComplaint']}")
                st.success(f"**Action Taken:** {result['Action']}")
            else:
                # Cache Miss -> Save new complaint to persistent storage
                new_entry = {"text": new_complaint}
                st.session_state.complaint_cache.append(new_entry)
                save_cache(st.session_state.complaint_cache)
                
                result = {
                    "Status": "Unique Complaint (Cache Miss)",
                    "SimilarityScore": float(highest_score),
                    "MatchedExistingComplaint": None,
                    "Action": "Create new triage ticket"
                }
                st.success(f"**Status:** {result['Status']}")
                st.info(f"**Similarity Score:** `{result['SimilarityScore']:.4f}`")
                st.success(f"**Action Taken:** {result['Action']} (Saved to `complaint_cache.json`)")

        # 2. Feedback Hook Section (Appends output data to feedback_log.csv if clicked)
        st.markdown("---")
        st.subheader("Feedback Hook")
        st.write("Did the system make a mistake on this classification or duplicate check?")
        if st.button("This was wrong"):
            feedback_data = pd.DataFrame([{
                "Timestamp": pd.Timestamp.now(),
                "Input_Text": new_complaint,
                "Status_Output": result["Status"],
                "Similarity_Score": result["SimilarityScore"]
            }])
            
            if not os.path.exists(FEEDBACK_FILE):
                feedback_data.to_csv(FEEDBACK_FILE, index=False)
            else:
                feedback_data.to_csv(FEEDBACK_FILE, mode='a', header=False, index=False)
                
            st.toast("Feedback logged successfully. Thank you!")

# Sidebar View of Stored Tickets
st.sidebar.header("Persistent Ticket Store")
st.sidebar.text(f"Total Cached Tickets: {len(st.session_state.complaint_cache)}")

if st.sidebar.button("Clear Persistent Cache"):
    st.session_state.complaint_cache = []
    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)
    st.sidebar.success("Cache cleared successfully!")
    st.rerun()

if st.session_state.complaint_cache:
    st.sidebar.markdown("---")
    for i, item in enumerate(st.session_state.complaint_cache, 1):
        st.sidebar.text(f"{i}. {item['text'][:40]}...")