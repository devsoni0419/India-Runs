# Redrob AI Intelligent Candidate Discovery & Ranking System

This repository contains the production-grade, highly optimized Candidate Discovery & Ranking System built for the Redrob Hackathon Challenge.

The system retrieves, ranks, and filters candidate profiles from a pool of 100,000 records to identify the **Top 100** candidates for a **Senior AI Engineer — Founding Team** role.

---

## 🚀 Key Features & Architectural Design

### 1. Honeypot & Trap Screening
The dataset contains subtly impossible "honeypots" designed to fool standard keyword-embedding vectors (e.g. candidates claiming 8 years of experience at a company founded 3 years ago, or listing expert skills with 0 months duration). 
- Our ranker programmatically identifies these impossible records and filters them out.
- We check for **inverted salary ranges** (`min > max` expected salary), which filters out over 18,000 spam profiles.
- We penalize candidate profiles that show **consulting/services company histories** (e.g. worked *only* at TCS, Infosys, Wipro, etc. with no product company background).
- We discount **keyword stuffers** (e.g. candidates whose current title is "HR Manager" or "Graphic Designer" but have stuffed their profiles with AI keywords).

### 2. Two-Stage Retrieval and Ranking
- **Stage 1: Semantic Retrieval**: Utilizes a highly optimized local embedding model (`BAAI/bge-small-en-v1.5`) running on CPU to embed candidate profiles and the Job Description. It indexes them via **FAISS** (with a vectorized Numpy dot product fallback) and retrieves the **Top 500** candidates in less than 0.05 seconds.
- **Stage 2: Hybrid Reranking**: Applies a recruiter-like scoring matrix to the Top 500 candidates. The final score is fused from:
  - **Semantic Score (40%)**: Cosine similarity from semantic embedding search.
  - **Skills Fit (20%)**: Core skills overlap weighted by proficiency and trust-verified by checking duration > 0.
  - **Experience Fit (15%)**: Alignment with the target 5-9 years experience band.
  - **Behavioral Signal (15%)**: Platform availability score (notice period <= 30 days, recent active login, high response rate, and active GitHub contribution).
  - **Education Pedigree (10%)**: Tiered scoring for computer science/engineering degrees from Tier-1 institutions.

### 3. Factual Explanation Engine
A template-based natural language generator builds 1-2 sentence summaries detailing exactly why a candidate is ranked at their position (e.g., matching actual skills, experience, location, notice period, and responsiveness) while ensuring maximum text variation and avoiding LLM hallucinations.

---

## 🛠 Setup & Installation

### Prerequisites
- Python 3.10+
- Recommended: 16 GB RAM and CPU with 4+ cores.

### Install Dependencies
Run the following command to install the required libraries:
```bash
pip install -r requirements.txt
```

---

## 🏃 Running the Ranker

To run the ranking pipeline end-to-end and generate the output `O(1) Squad.csv`, execute:
```bash
python rank.py --candidates ./candidates.jsonl --out "O(1) Squad.csv"
```

> [!NOTE]
> **Performance Optimization**: The first execution of `rank.py` downloads the BGE-small model and generates the vector embeddings for all 100,000 candidates (taking about 2-3 minutes on CPU). It then caches the embeddings locally as `candidate_embeddings.npy` and `candidate_ids.json`.
>
> On any subsequent runs, the pipeline detects the cache files and loads them instantly, completing the end-to-end ranking in **under 2 seconds**, which satisfies the strict 5-minute CPU constraint.

---

## 🔍 Validation

To validate your generated CSV file against the official challenge validation rules, run:
```bash
python validate_submission.py "O(1) Squad.csv"
```
This script validates:
1. Exact row count (header + 100 data rows).
2. Monotonically non-increasing scores.
3. Correct column ordering (`candidate_id,rank,score,reasoning`).
4. Correct formatting of candidate IDs and ranks.
5. Deterministic tie-breaking.

---

## 🛡️ Sandbox Local Application

To satisfy the **reproducibility requirement**, we provide a self-contained Streamlit application that can run locally. It allows running the ranking pipeline end-to-end on a small sample of candidates (or using the pre-loaded 100-candidate local dataset).

### Running the Sandbox:
1. Ensure all dependencies from `requirements.txt` are installed.
2. Launch the Streamlit server:
   ```bash
   streamlit run app.py
   ```
3. Open `http://localhost:8501` in your browser.
4. Select **Use Pre-loaded Local Sample** or upload a custom candidate subset (under 200MB) to run and download a validated rankings CSV.


#
