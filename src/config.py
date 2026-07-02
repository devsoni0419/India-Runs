from pathlib import Path

# Paths
BASE_DIR = Path("c:/Users/devso/recruiter india run")
DATA_DIR = BASE_DIR
OUTPUT_DIR = BASE_DIR
EMBEDDINGS_CACHE_FILE = BASE_DIR / "candidate_embeddings.npz"
CANDIDATE_IDS_CACHE_FILE = BASE_DIR / "candidate_ids.json"

CANDIDATES_FILE = DATA_DIR / "candidates.jsonl"
CANDIDATES_GZ_FILE = DATA_DIR / "candidates.jsonl.gz"

# Model Config
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384
BATCH_SIZE = 256

# Ranking Weight Config
WEIGHT_SEMANTIC = 0.40
WEIGHT_SKILLS = 0.20
WEIGHT_EXPERIENCE = 0.15
WEIGHT_BEHAVIOR = 0.15
WEIGHT_EDUCATION = 0.10

# Job Requirements
MIN_EXPERIENCE_YEARS = 5.0
MAX_EXPERIENCE_YEARS = 9.0

PREFERRED_LOCATIONS = ["noida", "pune", "delhi", "ncr", "bangalore", "bengaluru", "hyderabad", "mumbai"]

# Trap / Honeypot Filters
SERVICES_COMPANIES = [
    "tcs", "tata consultancy services", "infosys", "wipro", 
    "accenture", "cognizant", "capgemini", "hcl", "tech mahindra", "l&t"
]

NON_RELEVANT_TITLES = [
    "hr manager", "human resources", "content writer", "copywriter",
    "graphic designer", "ui/ux designer", "business analyst", 
    "marketing manager", "sales executive", "accountant", 
    "civil engineer", "mechanical engineer", "customer support",
    "operations manager", "project manager"
]

RELEVANT_TECH_KEYWORDS = [
    "ai", "ml", "machine learning", "deep learning", "nlp", "search", 
    "retrieval", "ranking", "recommendation", "data scientist", "nlp engineer",
    "applied ml", "backend", "software engineer", "full stack", "cloud", "data engineer"
]

CORE_AI_SKILLS = [
    "embeddings", "retrieval", "vector database", "faiss", "pinecone", 
    "qdrant", "weaviate", "milvus", "elasticsearch", "opensearch", 
    "python", "ndcg", "mrr", "map", "pytorch", "tensorflow", "scikit-learn", 
    "sentence-transformers", "hugging face", "llm", "large language models", 
    "fine-tuning", "lora", "qlora", "peft", "xgboost", "learning-to-rank",
    "recommendation systems", "natural language processing", "information retrieval"
]
