"""Task 2: extraction, chunking strategies, chunk metadata, stable IDs and incremental indexing."""
from pathlib import Path
from datetime import date
import hashlib, json, re, uuid
import fitz
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter

DOC_DIR = Path("documents")
OUT = Path("artifacts")
OUT.mkdir(exist_ok=True)

def extract_document(path):
    suffix = path.suffix.lower()
    pages = []
    if suffix == ".pdf":
        with fitz.open(path) as pdf:
            for page_no, page in enumerate(pdf, 1):
                pages.append((page_no, page.get_text("text")))
    elif suffix == ".docx":
        doc = Document(path)
        pages.append((1, "\n".join(p.text for p in doc.paragraphs)))
    elif suffix == ".txt":
        pages.append((1, path.read_text(encoding="utf-8")))
    else:
        raise ValueError(f"Unsupported file: {path}")
    return pages

def sentence_chunks(text, max_chars=700):
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, current = [], ""
    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return chunks

def fixed_chunks(text, size=512, overlap=50):
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    step = size - overlap
    return [text[i:i+size] for i in range(0, len(text), step) if text[i:i+size].strip()]

def recursive_chunks(text, size=512, overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_text(text)

def stable_id(source, page, chunk_index, content):
    # Content hash ensures an edited chunk gets a new ID; unchanged chunks keep their IDs.
    key = f"{source}|{page}|{chunk_index}|{hashlib.sha256(content.encode()).hexdigest()}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))

def make_chunks():
    records = []
    for path in sorted(DOC_DIR.iterdir()):
        if path.is_file() and path.suffix.lower() in {".pdf", ".docx", ".txt"}:
            for page, text in extract_document(path):
                for idx, chunk in enumerate(recursive_chunks(text)):
                    records.append({
                        "id": stable_id(path.name, page, idx, chunk),
                        "text": chunk, "source": path.name, "page": page,
                        "section": "General", "created_date": date.today().isoformat(),
                        "chunk_index": idx, "doc_type": path.suffix.lower().lstrip(".").upper()
                    })
    return records

def describe_chunk(chunk):
    return f"{chunk['source']} | page {chunk['page']} | section: {chunk['section']} | chunk {chunk['chunk_index']}"

def main():
    docs = []
    for path in sorted(DOC_DIR.iterdir()):
        if path.is_file() and path.suffix.lower() in {".pdf", ".docx", ".txt"}:
            docs.extend((path, page, text) for page, text in extract_document(path))
    if len({p.name for p,_,_ in docs}) < 5:
        raise RuntimeError("Place at least five supported documents in documents/")

    print("CHUNK COUNT COMPARISON (character-based demonstration)")
    for path, page, text in docs:
        fixed = fixed_chunks(text, 512, 50)
        sent = sentence_chunks(text, 512)
        rec = recursive_chunks(text, 512, 50)
        print(f"{path.name} page {page}: fixed={len(fixed)}, sentence={len(sent)}, recursive={len(rec)}")
    sample = " ".join("This is a sentence for demonstrating chunk boundaries." for _ in range(20))
    cut = fixed_chunks(sample, 100, 10)
    print("\nSentence cut in half example:")
    print("Chunk 1 ending:", cut[0][-35:])
    print("Chunk 2 beginning:", cut[1][:35])

    print("\nCHUNK SIZE EXPERIMENT")
    # Lightweight reproducible proxy: test whether each known question's answer phrase
    # appears in a chunk. Replace these questions/answers with your own document QA set.
    qa = [
        ("What is retrieval augmented generation?", "retrieval augmented generation"),
        ("What is an embedding?", "embedding"),
        ("What is cosine similarity?", "cosine similarity"),
        ("What is a vector database?", "vector database"),
        ("What is semantic search?", "semantic search"),
        ("What is an API?", "api"),
        ("What is metadata?", "metadata"),
        ("What is an index?", "index"),
        ("What is clustering?", "clustering"),
        ("What is an approximate nearest neighbour?", "approximate nearest")
    ]
    all_text = "\n".join(t for _,_,t in docs).lower()
    rows = []
    for size in (128, 256, 512, 1024):
        chunks = [c.lower() for _,_,t in docs for c in fixed_chunks(t, size, min(50, size//4))]
        hits = sum(any(answer in c for c in chunks) for _,answer in qa)
        rows.append({"chunk_size":size, "hits":hits, "questions":len(qa), "hit_rate":hits/len(qa)})
    print(json.dumps(rows, indent=2))

    records = make_chunks()
    (OUT / "chunks.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"\nSaved {len(records)} chunks to artifacts/chunks.json")
    if records:
        print("Citation:", describe_chunk(records[0]))

    # Stable-ID/incremental-index proof using an in-memory ID->content map.
    first = {r["id"]: r["text"] for r in records}
    second = {r["id"]: r["text"] for r in make_chunks()}
    print(f"Run 1: {len(first)} points | Run 2: {len(second)} points | unchanged IDs: {len(first.keys() & second.keys())}")
    if records:
        edited = dict(records[0])
        edited["text"] += " Added one edited sentence."
        edited["id"] = stable_id(edited["source"], edited["page"], edited["chunk_index"], edited["text"])
        third = dict(second)
        third.pop(records[0]["id"], None)
        third[edited["id"]] = edited["text"]
        print(f"Run 3 (one chunk edited): {len(third)} points | replaced: {len(first.keys() - third.keys())}")

if __name__ == "__main__":
    main()
