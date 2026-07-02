import streamlit as st
import pandas as pd
import json
import os
import sys
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.parser import is_honeypot, build_candidate_representation
from src.embedder import CandidateEmbedder
from src.retriever import CandidateRetriever
from src.ranker import CandidateRanker
from src.explainer import CandidateExplainer

# Default JD query matching target profile
JD_DEFAULT = (
    "Senior AI Engineer Founding Team. Deployed embeddings-based retrieval systems, "
    "sentence-transformers, vector databases like FAISS, Pinecone, Qdrant, Milvus. "
    "Strong Python, evaluation frameworks NDCG, MRR, MAP, offline evaluation. "
    "5-9 years of experience, product company background, Noida Pune."
)

st.set_page_config(
    page_title="Redrob Sandbox App",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Redrob Candidate Ranking Sandbox")
st.write("This sandbox runs the candidate ranking pipeline on the pre-loaded 100 candidate sample dataset (`sample_candidates.jsonl`) to demonstrate reproducibility within the compute budget.")

# Load pre-loaded candidate dataset
candidates = []
local_path = Path("./sample_candidates.jsonl")

if local_path.exists():
    st.success("Loaded pre-loaded candidate dataset (100 sample records) successfully.")
    try:
        from src.parser import iter_candidates
        for cand in iter_candidates(local_path):
            candidates.append(cand)
    except Exception as e:
        st.error(f"Error reading sample candidate file: {e}")
        st.stop()
else:
    st.error("Error: `sample_candidates.jsonl` not found. Please make sure the sample dataset is generated.")
    st.stop()

# Job Description Input
st.subheader("📋 Target Job Description")
jd_query = st.text_area(
    "Search query reflecting the key job description requirements:",
    value=JD_DEFAULT,
    height=120
)

# Run Button
if st.button("🚀 Run Ranking & Generate CSV"):
    with st.spinner("Executing pipeline end-to-end (Embedding retrieval + Score fusion reranking)..."):
        total_records = len(candidates)
        
        # Filter Honeypots
        clean_candidates = []
        honeypot_count = 0
        for cand in candidates:
            if is_honeypot(cand):
                honeypot_count += 1
            else:
                clean_candidates.append(cand)
                
        # Compute Embeddings dynamically
        embedder = CandidateEmbedder()
        texts = [build_candidate_representation(cand) for cand in clean_candidates]
        embeddings = embedder.embed_texts(texts)
        candidate_ids = [c["candidate_id"] for c in clean_candidates]
        id_to_index = {cid: idx for idx, cid in enumerate(candidate_ids)}
        
        # Embed Query
        query_emb = embedder.embed_query(jd_query)
        
        # Retrieve Stage 1
        retriever = CandidateRetriever()
        retriever.build_index(embeddings, candidate_ids)
        retrieved = retriever.retrieve(query_emb, top_k=len(clean_candidates))
        
        # Rerank Stage 2
        ranker = CandidateRanker()
        scored_candidates = []
        
        for item in retrieved:
            cid = item["candidate_id"]
            sem_score = item["semantic_score"]
            cand_idx = id_to_index[cid]
            cand = clean_candidates[cand_idx]
            
            final_score, reason, components = ranker.score_candidate(cand, sem_score)
            if final_score > 0.0:
                scored_candidates.append({
                    "candidate": cand,
                    "candidate_id": cid,
                    "score": final_score,
                    "components": components
                })
                
        # Sort and Tie-break
        scored_candidates.sort(key=lambda x: (-round(x["score"], 4), x["candidate_id"]))
        
        # Generate Explanations
        top_count = min(100, len(scored_candidates))
        top_selection = scored_candidates[:top_count]
        
        explainer = CandidateExplainer()
        final_rows = []
        
        for idx, item in enumerate(top_selection):
            rank = idx + 1
            cid = item["candidate_id"]
            score = item["score"]
            cand = item["candidate"]
            components = item["components"]
            
            reasoning = explainer.generate_explanation(rank, score, cand, components)
            
            final_rows.append({
                "candidate_id": cid,
                "rank": rank,
                "score": round(score, 4),
                "reasoning": reasoning
            })
            
        # Create Results DataFrame
        results_df = pd.DataFrame(final_rows)
        
        # Display Summary metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Candidates", total_records)
        col2.metric("Honeypots Screened", honeypot_count)
        col3.metric("Successfully Ranked", len(results_df))
        
        st.subheader("🏆 Ranked Candidates Output")
        st.dataframe(results_df, use_container_width=True)
        
        # Download Button
        csv_data = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download submission.csv",
            data=csv_data,
            file_name="submission.csv",
            mime="text/csv"
        )
        st.success("Ranking pipeline successfully executed!")
