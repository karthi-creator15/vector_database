"""Task 4: 15-query retrieval evaluation and two-configuration comparison."""
import json
from pathlib import Path
import pandas as pd
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

QUERIES = [
    {"query":"What are embeddings used for?", "expected_source":"01_embeddings.pdf", "expected_page":1},
    {"query":"Explain semantic search", "expected_source":"01_embeddings.pdf", "expected_page":1},
    {"query":"What does cosine similarity measure?", "expected_source":"01_embeddings.pdf", "expected_page":1},
    {"query":"How do vector databases store vectors?", "expected_source":"02_vector_search.docx", "expected_page":1},
    {"query":"What is approximate nearest neighbour search?", "expected_source":"02_vector_search.docx", "expected_page":1},
    {"query":"Explain HNSW graph indexing", "expected_source":"02_vector_search.docx", "expected_page":1},
    {"query":"Why split documents into chunks?", "expected_source":"03_chunking.txt", "expected_page":1},
    {"query":"What is chunk overlap?", "expected_source":"03_chunking.txt", "expected_page":1},
    {"query":"Why attach metadata to chunks?", "expected_source":"03_chunking.txt", "expected_page":1},
    {"query":"What is a REST API?", "expected_source":"04_fastapi.pdf", "expected_page":1},
    {"query":"How does API key authentication work?", "expected_source":"04_fastapi.pdf", "expected_page":1},
    {"query":"What does a health endpoint do?", "expected_source":"04_fastapi.pdf", "expected_page":1},
    {"query":"What is retrieval augmented generation?", "expected_source":"05_rag.docx", "expected_page":1},
    {"query":"How does retrieval provide context to a language model?", "expected_source":"05_rag.docx", "expected_page":1},
    {"query":"What are limitations of RAG?", "expected_source":"05_rag.docx", "expected_page":1}
]

def evaluate(client, model, chunk_size, limit=5):
    hits1 = hits3 = hits5 = 0
    reciprocal_ranks = []
    for item in QUERIES:
        qv = model.encode([item["query"]], normalize_embeddings=True)[0].tolist()
        results = client.query_points(collection_name="week6_documents", query=qv, limit=limit).points
        rank = None
        for i, result in enumerate(results, 1):
            p = result.payload or {}
            if p.get("source") == item["expected_source"] and p.get("page") == item["expected_page"]:
                rank = i
                break
        hits1 += rank == 1
        hits3 += rank is not None and rank <= 3
        hits5 += rank is not None and rank <= 5
        reciprocal_ranks.append(1/rank if rank else 0)
    n = len(QUERIES)
    return {"config":f"chunk_{chunk_size}", "hit@1":hits1/n, "hit@3":hits3/n,
            "hit@5":hits5/n, "MRR":sum(reciprocal_ranks)/n}

def main():
    client = QdrantClient(path="./qdrant_data")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    # Config A/B values are measured against the currently indexed collection only.
    # Re-index with CHUNK_SIZE=256 or 512 before running each config for a fair comparison.
    size = int(__import__("os").getenv("CHUNK_SIZE", "512"))
    result = evaluate(client, model, size)
    print(pd.DataFrame([result]).to_string(index=False))
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/evaluation_results.json").write_text(json.dumps({"queries":QUERIES,"results":[result]}, indent=2))
    print("For a fair A/B test, index once at 256 and save results, then index at 512 and compare.")
    client.close()

if __name__ == "__main__":
    main()
