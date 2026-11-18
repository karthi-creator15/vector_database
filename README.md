# Week 6 – RAG Assessment (Qdrant + Ollama)

A Windows-friendly, four-task implementation covering embeddings, chunking, stable IDs,
FAISS ANN/HNSW, Qdrant vector storage, FastAPI search and retrieval evaluation.

## 1. Requirements

- Windows 10/11, Python 3.10–3.12 recommended
- Ollama (optional for Task 1 API embedding comparison)
- Internet for first-time Python/model downloads
- For local Qdrant, no Docker is required

> `llama3.2` is a text-generation model. For embedding API comparison use
> `nomic-embed-text` with Ollama. The retrieval API uses local `all-MiniLM-L6-v2`
> (384 dimensions), matching the Qdrant `vector(384)` collection.

## 2. Windows setup

Open PowerShell in this project folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

If PowerShell blocks activation, run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Ollama (Task 1 API comparison)

Install Ollama for Windows from https://ollama.com/download/windows, then:

```powershell
ollama pull nomic-embed-text
ollama list
```

Keep Ollama running. Optional text-generation model:
```powershell
ollama pull llama3.2
```

## 3. Sample documents

Five sample documents are included in `documents/` (PDF, DOCX and TXT).
You can replace or add documents. Task 2 extracts PDF pages with PyMuPDF,
DOCX paragraphs with python-docx, and TXT as plain text.

## 4. Run the tasks in order

```powershell
python task1_embeddings.py
python task2_chunking.py
python task3_vector_storage.py
```

Task 1 prints the hand-implemented cosine checks, sklearn comparison, embedding dimensions
and timing, similarity matrix, top pairs, semantic search, duplicate candidates and K-Means.

Task 2 prints per-document chunk strategy counts, a chunk-boundary example, chunk-size
experiment table, metadata citations and stable-ID incremental-index proof. Its default
chunking uses RecursiveCharacterTextSplitter with 512 characters and 50 overlap.
The size experiment is a small baseline proxy; for your submission, write 10 question/answer
pairs grounded in the actual documents and verify their expected pages.

Task 3 creates a local Qdrant collection with cosine distance and HNSW, embeds all chunks,
re-upserts identical IDs, runs payload-filtered search, and deletes one source. It also
compares FAISS exact search against HNSW on 50,000 random 128-D vectors.

## 5. Search API

Run Task 2 and Task 3 first to populate Qdrant. Set a private API key in `.env`.
Then:

```powershell
uvicorn app:app --reload
```

Health:
```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Search (replace the key with your `.env` API_KEY):
```powershell
$headers = @{"X-API-Key"="change-this-week6-key"}
$body = @{query="What is semantic search?"; top_k=5; doc_type="PDF"; min_score=0.2} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/search -Headers $headers -ContentType "application/json" -Body $body
```

Upload and index:
```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/documents -Headers $headers -Form @{file=Get-Item ".\documents\03_chunking.txt"}
```

Open interactive docs at http://127.0.0.1:8000/docs.

## 6. Evaluation

```powershell
python evaluate.py
```

`evaluate.py` contains 15 query/expected-source/page examples and calculates hit@1, hit@3,
hit@5 and MRR. For a valid two-configuration comparison, run Task 2 with chunk sizes 256
and 512 in separate clean Qdrant collections, then record each measured result in
`artifacts/evaluation_results.json`. Do not claim sample/example metrics as measured results.
The expected source names correspond to the supplied sample files.

Definitions:
- hit@k: fraction of queries with a relevant source/page in the first k results.
- reciprocal rank: 1/rank of the first relevant result, or 0 if absent.
- MRR: mean reciprocal rank over all evaluation queries.

## 7. Tests

With the API dependencies installed:

```powershell
pytest -v
```

Seven tests cover health, valid search, API-key protection, top_k lower/upper bounds,
empty query validation and document-type filtering.

## 8. Architecture

```text
PDF / DOCX / TXT
      |
      v
PyMuPDF / python-docx / text reader
      |
      v
Recursive chunking + metadata + deterministic IDs
      |
      v
all-MiniLM-L6-v2 (384-D embeddings)
      |
      v
Qdrant (Cosine + HNSW + payload filters)
      |
      v
FastAPI /search  ---> hit@k and MRR evaluation
```

## 9. Limitations and interpretation

- The local MiniLM model and Ollama embedding model have different dimensions; do not mix
  vectors in one Qdrant collection.
- The fixed-size demonstration in Task 2 is character-based to make boundaries visible.
  Token-aware splitting can be substituted for strict token-count experiments.
- The ten-question chunk-size experiment is a starter harness; use questions and answer
  spans verified against your own documents before reporting final hit rates.
- The Qdrant demo deletes the first source to demonstrate deletion; re-run Task 2 and Task 3
  to rebuild the full collection.
- Similarity is not proof of factual correctness. Validate retrieval with labelled queries.
- ANN timing varies by CPU, RAM, FAISS build, and system load; report your measured values.

## 10. Five-minute demo outline

1. Show the five source documents and run Task 2.
2. Explain metadata and stable IDs; rerun to show count remains stable.
3. Run Task 3; compare exact vs HNSW recall and inspect Qdrant filtering.
4. Start FastAPI, call `/health`, search, then upload a new file.
5. Run pytest and show hit@k/MRR evaluation.

## 11. Submission checklist

- [ ] Run all four tasks on your laptop and retain actual console outputs.
- [ ] Replace the starter QA pairs with ten verified document questions.
- [ ] Run both chunk-size configurations and save measured hit@k/MRR.
- [ ] Add your screenshots, GitHub repository URL and merged PR details.
- [ ] Do not copy the illustrative timing values from the assessment brief as your own results.
