"""Task 3: FAISS exact-vs-HNSW experiment and Qdrant vector storage."""
import time
import numpy as np
from pathlib import Path
import json
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

def ann_experiment():
    import faiss
    rng = np.random.default_rng(42)
    vectors = rng.random((50000, 128), dtype=np.float32)
    query = rng.random((1, 128), dtype=np.float32)
    exact = faiss.IndexFlatL2(128)
    exact.add(vectors)
    t0 = time.perf_counter()
    _, exact_ids = exact.search(query, 10)
    exact_ms = (time.perf_counter()-t0)*1000

    hnsw = faiss.IndexHNSWFlat(128, 32)
    hnsw.hnsw.efConstruction = 100
    hnsw.hnsw.efSearch = 64
    hnsw.add(vectors)
    t0 = time.perf_counter()
    _, approx_ids = hnsw.search(query, 10)
    hnsw_ms = (time.perf_counter()-t0)*1000
    recall = len(set(exact_ids[0]) & set(approx_ids[0])) / 10
    print(f"Exact: {exact_ms:.2f} ms | HNSW: {hnsw_ms:.2f} ms | recall@10: {recall:.2f}")
    return exact_ms, hnsw_ms, recall

def qdrant_demo():
    chunks_path = Path("artifacts/chunks.json")
    if not chunks_path.exists():
        raise RuntimeError("Run python task2_chunking.py first.")
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = QdrantClient(path="./qdrant_data")
    collection = "week6_documents"
    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(
        collection_name=collection,
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
        hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100)
    )
    texts = [c["text"] for c in chunks]
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    points = [
        models.PointStruct(id=c["id"], vector=v.tolist(), payload={k:c[k] for k in ("text","source","page","section","created_date","chunk_index","doc_type")})
        for c,v in zip(chunks, vectors)
    ]
    client.upsert(collection_name=collection, points=points, wait=True)
    print("Qdrant collection info:", client.get_collection(collection))
    # Re-upsert same point IDs: count remains stable.
    client.upsert(collection_name=collection, points=points, wait=True)
    print("After re-upsert count:", client.count(collection, exact=True).count)

    query_vec = model.encode(["vector search and retrieval"], normalize_embeddings=True)[0].tolist()
    result = client.query_points(
        collection_name=collection, query=query_vec, limit=5,
        query_filter=models.Filter(must=[
            models.FieldCondition(key="doc_type", match=models.MatchValue(value="PDF")),
            models.FieldCondition(key="page", range=models.Range(gte=3))
        ])
    ).points
    print(f"Filtered search (doc_type=PDF, page>=3): {len(result)} results")
    for p in result:
        print(p.score, p.payload["source"], p.payload["page"], p.payload["text"][:100])

    # Delete all points for one source using a payload filter.
    if chunks:
        source = chunks[0]["source"]
        client.delete(collection_name=collection, points_selector=models.FilterSelector(
            filter=models.Filter(must=[models.FieldCondition(key="source", match=models.MatchValue(value=source))])
        ), wait=True)
        print(f"Deleted source={source}; remaining={client.count(collection, exact=True).count}")
    client.close()

def main():
    print("ANN / HNSW: 5 LINES")
    print("1. ANN finds near neighbours quickly without comparing every vector.")
    print("2. Exact search checks all vectors and is a useful accuracy baseline.")
    print("3. HNSW builds a hierarchical navigable small-world graph.")
    print("4. Search greedily navigates upper layers, then refines at lower layers.")
    print("5. M, efConstruction and efSearch trade memory/build/search time for recall.")
    ann_experiment()
    qdrant_demo()
    print("\nPGVECTOR VS QDRANT")
    print("| Aspect | pgvector | Qdrant |")
    print("| Setup | PostgreSQL extension | Dedicated vector service/local mode |")
    print("| Filtering | SQL WHERE | Payload filters |")
    print("| Scale | Strong when vectors live with relational data | Purpose-built vector workloads |")
    print("| Choose when | Existing PostgreSQL and SQL joins matter | Vector-first search and simple payload filtering matter |")

if __name__ == "__main__":
    main()
