import numpy as np
from src.config import EMBEDDING_DIM
from src.logger import setup_logger

logger = setup_logger()

try:
    import faiss
    FAISS_AVAILABLE = True
    logger.info("FAISS library is successfully imported.")
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS is not installed. Falling back to Numpy vector search.")

class CandidateRetriever:
    def __init__(self, embedding_dim=EMBEDDING_DIM):
        self.embedding_dim = embedding_dim
        self.index = None
        self.candidate_ids = []
        self.embeddings = None

    def build_index(self, embeddings, candidate_ids):
        """
        Builds the retrieval index using FAISS or Numpy matrix fallback.
        """
        self.candidate_ids = list(candidate_ids)
        self.embeddings = np.array(embeddings, dtype=np.float32)

        if FAISS_AVAILABLE:
            logger.info("Building FAISS Flat Inner Product (IP) Index...")
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.index.add(self.embeddings)
            logger.info("FAISS Index built successfully.")
        else:
            logger.info("Numpy matrix search initialized.")

    def retrieve(self, query_embedding, top_k=500):
        """
        Performs fast similarity search to retrieve the top_k candidate IDs and their similarity scores.
        """
        query_vector = np.array(query_embedding, dtype=np.float32).reshape(1, -1)

        if FAISS_AVAILABLE and self.index is not None:
            # FAISS returns (distances, indices)
            scores, indices = self.index.search(query_vector, top_k)
            scores = scores[0]
            indices = indices[0]
        else:
            # Numpy dot product (embeddings are normalized, so dot product is Cosine Similarity)
            # shape: (100000,)
            sims = np.dot(self.embeddings, query_vector[0])
            indices = np.argsort(sims)[::-1][:top_k]
            scores = sims[indices]

        results = []
        for rank, (score, idx) in enumerate(zip(scores, indices)):
            # Ensure index is within range
            if idx < len(self.candidate_ids):
                results.append({
                    "candidate_id": self.candidate_ids[idx],
                    "semantic_score": float(score),
                    "idx": int(idx)
                })
        return results
