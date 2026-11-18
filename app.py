"""Task 4: FastAPI semantic search API backed by local Qdrant."""
import os, uuid, tempfile
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, Depends, Header, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from task2_chunking import extract_document, recursive_chunks, stable_id

load_dotenv()
COLLECTION = os.getenv("QDRANT_COLLECTION", "week6_documents")
API_KEY = os.getenv("API_KEY", "change-this-week6-key")
QDRANT_PATH = os.getenv("QDRANT_PATH", "./qdrant_data")
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = SentenceTransformer(MODEL_NAME)
    app.state.client = QdrantClient(path=QDRANT_PATH)
    if not app.state.client.collection_exists(COLLECTION):
        app.state.client.create_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
            hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100)
        )
    yield
    app.state.client.close()

app = FastAPI(title="Week 6 RAG Search API", version="1.0.0", lifespan=lifespan)

def verify_key(x_api_key: str = Header(default="")):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    doc_type: Optional[str] = None
    min_score: Optional[float] = Field(default=None, ge=0, le=1)

@app.get("/health")
def health():
    return {"status":"ok"}

@app.post("/search")
def search(request: SearchRequest, _: None = Depends(verify_key)):
    model, client = app.state.model, app.state.client
    vector = model.encode([request.query], normalize_embeddings=True)[0].tolist()
    conditions = []
    if request.doc_type:
        conditions.append(models.FieldCondition(key="doc_type", match=models.MatchValue(value=request.doc_type.upper())))
    query_filter = models.Filter(must=conditions) if conditions else None
    hits = client.query_points(
        collection_name=COLLECTION, query=vector, limit=request.top_k,
        query_filter=query_filter, with_payload=True
    ).points
    results = []
    for hit in hits:
        if request.min_score is not None and hit.score < request.min_score:
            continue
        p = hit.payload or {}
        results.append({
            "source":p.get("source"), "page":p.get("page"),
            "section":p.get("section"), "score":float(hit.score),
            "text":p.get("text")
        })
    return {"query":request.query, "results":results}

@app.post("/documents")
async def upload_document(file: UploadFile = File(...), _: None = Depends(verify_key)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".pdf", ".docx", ".txt"}:
        raise HTTPException(400, "Only PDF, DOCX and TXT are supported")
    content = await file.read()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)
    try:
        pages = extract_document(tmp_path)
        points = []
        for page, text in pages:
            for idx, chunk in enumerate(recursive_chunks(text, 512, 50)):
                source = file.filename or tmp_path.name
                point_id = stable_id(source, page, idx, chunk)
                vector = app.state.model.encode([chunk], normalize_embeddings=True)[0].tolist()
                points.append(models.PointStruct(id=point_id, vector=vector, payload={
                    "text":chunk, "source":source, "page":page, "section":"General",
                    "created_date":__import__("datetime").date.today().isoformat(),
                    "chunk_index":idx, "doc_type":suffix[1:].upper()
                }))
        if points:
            app.state.client.upsert(collection_name=COLLECTION, points=points, wait=True)
        return {"filename":file.filename, "indexed_chunks":len(points)}
    finally:
        tmp_path.unlink(missing_ok=True)
