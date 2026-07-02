import argparse
import sys
import os
import csv
from pathlib import Path

# Setup paths
sys.path.append(str(Path(__file__).parent))

from src.config import CANDIDATES_FILE, CANDIDATES_GZ_FILE, EMBEDDINGS_CACHE_FILE
from src.logger import setup_logger
from src.parser import iter_candidates, is_honeypot
from src.embedder import CandidateEmbedder
from src.retriever import CandidateRetriever
from src.ranker import CandidateRanker
from src.explainer import CandidateExplainer

logger = setup_logger()

# Search query capturing the JD semantic essence
JD_QUERY = (
    "Senior AI Engineer Founding Team. Deployed embeddings-based retrieval systems, "
    "sentence-transformers, vector databases like FAISS, Pinecone, Qdrant, Milvus. "
    "Strong Python, evaluation frameworks NDCG, MRR, MAP, offline evaluation. "
    "5-9 years of experience, product company background, Noida Pune."
)

def run_ranking(candidates_path: str, output_path: str):
    logger.info("=========================================")
    logger.info("Starting Redrob AI Candidate Ranker v4")
    logger.info("=========================================")
    
    # 1. Resolve candidates file
    cands_path = Path(candidates_path)
    if not cands_path.exists():
        # Fallback check
        if cands_path.name == "candidates.jsonl" and Path(cands_path.parent / "candidates.jsonl.gz").exists():
            cands_path = Path(cands_path.parent / "candidates.jsonl.gz")
            logger.info(f"candidates.jsonl not found, using gzipped candidate pool: {cands_path}")
        else:
            logger.error(f"Candidates file not found at {candidates_path}")
            sys.exit(1)
            
    # Load all candidates into a list (or check cache first to save memory if precomputed)
    # Since we need to get full JSONs for reranking, we'll keep a map of ID -> candidate details
    # To save memory, we can index them in memory, but 100k records fits comfortably in < 1GB RAM.
    logger.info(f"Loading candidates from {cands_path}...")
    candidates = []
    for cand in iter_candidates(cands_path):
        candidates.append(cand)
    logger.info(f"Loaded {len(candidates)} candidates.")

    # 2. Embedding generation
    embedder = CandidateEmbedder()
    embeddings, candidate_ids = embedder.load_or_compute_embeddings(candidates)
    
    # Map candidate ID to index for O(1) lookup
    id_to_index = {cid: idx for idx, cid in enumerate(candidate_ids)}
    
    # 3. Retrieve query embedding
    logger.info("Encoding Job Description query...")
    query_emb = embedder.embed_query(JD_QUERY)
    
    # 4. Search & Retrieve Top 500
    retriever = CandidateRetriever()
    retriever.build_index(embeddings, candidate_ids)
    
    logger.info("Retrieving top 500 candidates via semantic search...")
    retrieved = retriever.retrieve(query_emb, top_k=500)
    logger.info(f"Retrieved {len(retrieved)} candidates.")
    
    # 5. Hybrid Reranking on Top 500
    ranker = CandidateRanker()
    scored_candidates = []
    
    for item in retrieved:
        cid = item["candidate_id"]
        sem_score = item["semantic_score"]
        
        # Look up candidate details
        cand_idx = id_to_index[cid]
        cand = candidates[cand_idx]
        
        final_score, reason, components = ranker.score_candidate(cand, sem_score)
        
        # Don't rank flagged honeypots or disqualified candidates if they get 0 score
        if final_score > 0.0:
            scored_candidates.append({
                "candidate": cand,
                "candidate_id": cid,
                "score": final_score,
                "components": components
            })
            
    # 6. Sorting and Tie-breaking
    # Sort by rounded score descending, then candidate_id ascending (deterministic tie-breaking)
    scored_candidates.sort(key=lambda x: (-round(x["score"], 4), x["candidate_id"]))
    
    # Select Top 100
    top_100 = scored_candidates[:100]
    logger.info(f"Completed hybrid ranking. Selected top {len(top_100)} candidates.")
    
    # 7. Generate Explanations
    explainer = CandidateExplainer()
    final_rows = []
    
    for i, item in enumerate(top_100):
        rank = i + 1
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
        
    # Write to CSV
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Writing ranked list to {out_path}...")
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        for row in final_rows:
            writer.writerow(row)
            
    logger.info(f"Ranker pipeline successfully completed! Output saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Redrob AI Candidate Ranker Pipeline")
    parser.add_argument(
        "--candidates", 
        type=str, 
        default="./candidates.jsonl", 
        help="Path to the candidate dataset (.jsonl or .jsonl.gz)"
    )
    parser.add_argument(
        "--out", 
        type=str, 
        default="./submission.csv", 
        help="Path to save the output ranking CSV"
    )
    args = parser.parse_args()
    
    run_ranking(args.candidates, args.out)
