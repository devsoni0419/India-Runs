import numpy as np
import json
import os
from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL_NAME, EMBEDDINGS_CACHE_FILE, CANDIDATE_IDS_CACHE_FILE, BATCH_SIZE
from src.logger import setup_logger

logger = setup_logger()

class CandidateEmbedder:
    def __init__(self, model_name=EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self.model = None

    def _load_model(self):
        if self.model is None:
            logger.info(f"Loading embedding model: {self.model_name}...")
            import torch
            
            # Determine the best available hardware accelerator
            device_str = "cpu"
            dml_device = None
            
            if torch.cuda.is_available():
                device_str = "cuda"
                logger.info("NVIDIA GPU detected. Using CUDA.")
            elif hasattr(torch, "xpu") and torch.xpu.is_available():
                device_str = "xpu"
                logger.info("Intel GPU (XPU) detected. Using XPU backend.")
            else:
                try:
                    import torch_directml
                    if torch_directml.is_available():
                        dml_device = torch_directml.device()
                        logger.info("Intel GPU (DirectML) detected. Using DirectML backend.")
                except ImportError:
                    pass

            # Load model
            if dml_device is not None:
                # Load on CPU first, then move to DirectML device
                self.model = SentenceTransformer(self.model_name, device="cpu")
                self.model.to(dml_device)
                logger.info("Embedding model loaded successfully on DirectML device.")
            else:
                # For CPU, configure optimal thread count
                if device_str == "cpu":
                    num_threads = min(os.cpu_count(), 8) if os.cpu_count() else 4
                    torch.set_num_threads(num_threads)
                    logger.info(f"Configured PyTorch with {num_threads} CPU threads.")
                
                self.model = SentenceTransformer(self.model_name, device=device_str)
                logger.info(f"Embedding model loaded successfully on {device_str.upper()}.")

    def embed_texts(self, texts, batch_size=BATCH_SIZE):
        self._load_model()
        # Instruct BGE queries/documents
        # BGE small requires query instruction for retrieval, but for document embedding we don't need prefixes.
        logger.info(f"Generating embeddings for {len(texts)} text blocks on CPU...")
        embeddings = self.model.encode(
            texts, 
            batch_size=batch_size, 
            show_progress_bar=True, 
            normalize_embeddings=True
        )
        return np.array(embeddings, dtype=np.float32)

    def embed_query(self, query_text):
        self._load_model()
        # BGE-small query format prefix: "Represent this sentence for searching relevant passages: "
        query_text_formatted = f"Represent this sentence for searching relevant passages: {query_text}"
        embedding = self.model.encode(
            query_text_formatted, 
            normalize_embeddings=True
        )
        return np.array(embedding, dtype=np.float32)

    def load_or_compute_embeddings(self, candidates):
        """
        Loads precomputed embeddings from disk if they exist,
        otherwise computes them and saves them to the cache.
        """
        if os.path.exists(EMBEDDINGS_CACHE_FILE) and os.path.exists(CANDIDATE_IDS_CACHE_FILE):
            logger.info("Found cached candidate embeddings. Loading from cache...")
            if str(EMBEDDINGS_CACHE_FILE).endswith(".npz"):
                with np.load(EMBEDDINGS_CACHE_FILE) as data:
                    embeddings = data["embeddings"].astype(np.float32)
            else:
                embeddings = np.load(EMBEDDINGS_CACHE_FILE)
            with open(CANDIDATE_IDS_CACHE_FILE, "r") as f:
                candidate_ids = json.load(f)
            logger.info(f"Loaded {len(embeddings)} cached embeddings successfully.")
            return embeddings, candidate_ids

        logger.info("Cached embeddings not found. Re-computing embeddings from candidates pool...")
        candidate_ids = []
        texts = []
        for cand in candidates:
            candidate_ids.append(cand["candidate_id"])
            # build textual representation
            from src.parser import build_candidate_representation
            texts.append(build_candidate_representation(cand))
            
        embeddings = self.embed_texts(texts)
        
        # Save to cache
        logger.info(f"Saving computed embeddings to cache at {EMBEDDINGS_CACHE_FILE}...")
        if str(EMBEDDINGS_CACHE_FILE).endswith(".npz"):
            np.savez_compressed(EMBEDDINGS_CACHE_FILE, embeddings=embeddings.astype(np.float16))
        else:
            np.save(EMBEDDINGS_CACHE_FILE, embeddings)
        with open(CANDIDATE_IDS_CACHE_FILE, "w") as f:
            json.dump(candidate_ids, f)
        logger.info("Cache saved successfully.")
        
        return embeddings, candidate_ids
