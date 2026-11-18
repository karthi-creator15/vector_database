import time
from itertools import combinations
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
from sklearn.cluster import KMeans

SENTENCES = [
    "A dog is playing in the garden.",
    "A puppy runs around outside.",
    "The cat is sleeping on the sofa.",
    "A kitten rests on a couch.",
    "How can I reset my account password?",
    "I forgot my login password and need to change it.",
    "The weather is sunny and warm today.",
    "It is a bright, pleasant day with sunshine.",
    "A spacecraft is exploring a distant planet.",
    "A rocket travels through space toward Mars.",
]

def cosine_similarity(a, b):
    """Compute cosine similarity without sklearn."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("Vectors must have the same shape")
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def ollama_embeddings(texts, model="nomic-embed-text"):
    import ollama
    response = ollama.embed(model=model, input=texts)
    return np.asarray(response["embeddings"], dtype=np.float32)

def main():
    print("COSINE UNIT TESTS")
    tests = [([1,2,3], [2,4,6], 1.0), ([1,0], [0,1], 0.0), ([1,0], [-1,0], -1.0)]
    for a, b, expected in tests:
        actual = cosine_similarity(a, b)
        print(f"{a} vs {b} -> {actual:.1f} (expected {expected:.1f})")
        assert np.isclose(actual, expected)
        assert np.isclose(actual, sklearn_cosine([a], [b])[0][0])

    model = SentenceTransformer("all-MiniLM-L6-v2")
    start = time.perf_counter()
    local_vectors = model.encode(SENTENCES, normalize_embeddings=True, show_progress_bar=False)
    local_time = time.perf_counter() - start
    print(f"\nLocal all-MiniLM-L6-v2: shape={local_vectors.shape}, time={local_time:.3f}s")

    try:
        start = time.perf_counter()
        api_vectors = ollama_embeddings(SENTENCES)
        api_time = time.perf_counter() - start
        print(f"Ollama nomic-embed-text: shape={api_vectors.shape}, time={api_time:.3f}s")
    except Exception as exc:
        api_vectors, api_time = None, None
        print(f"Ollama comparison skipped: {exc}")
        print("Start Ollama and run: ollama pull nomic-embed-text")

    matrix = local_vectors @ local_vectors.T
    print("\nSimilarity matrix (local model):")
    print(np.round(matrix, 3))
    pairs = sorted(
        [(float(matrix[i,j]), i, j) for i,j in combinations(range(len(SENTENCES)), 2)],
        reverse=True
    )
    print("\nTop 3 similar pairs:")
    for score, i, j in pairs[:3]:
        print(f"{score:.3f}: [{i}] {SENTENCES[i]} <-> [{j}] {SENTENCES[j]}")

    print("\nNo-common-word semantic pair:")
    a, b = SENTENCES[0], SENTENCES[1]
    print(f"{a!r} <-> {b!r}: similarity={matrix[0,1]:.3f}")
    print("(The wording differs; the sentences express a closely related meaning.)")

    queries = ["puppy playing outdoors", "I need help changing my password", "space travel to Mars"]
    print("\nSEMANTIC SEARCH")
    for query in queries:
        q = model.encode([query], normalize_embeddings=True)[0]
        scores = local_vectors @ q
        idx = int(np.argmax(scores))
        print(f"Query: {query}\n  Match: {SENTENCES[idx]} ({scores[idx]:.3f})")

    print("\nDUPLICATE DETECTION (threshold=0.72)")
    threshold = 0.72
    for score, i, j in pairs:
        if score >= threshold:
            print(f"Possible duplicate ({score:.3f}): {SENTENCES[i]} | {SENTENCES[j]}")

    print("\nK-MEANS CLUSTERING")
    labels = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(local_vectors)
    keyword_names = [
        ("Animals", {"dog", "puppy", "cat", "kitten", "garden", "sofa", "couch"}),
        ("Account support", {"account", "password", "login", "reset", "forgot"}),
        ("Weather", {"weather", "sunny", "warm", "bright", "day", "sunshine"}),
        ("Space", {"spacecraft", "rocket", "planet", "space", "mars"})
    ]
    for label in sorted(set(labels)):
        members = [SENTENCES[i] for i in range(len(SENTENCES)) if labels[i] == label]
        words = set(" ".join(members).lower().replace(".", "").split())
        name = max(keyword_names, key=lambda item: len(words & item[1]))[0]
        print(f"Cluster {label} ({name}): {members}")

    print("\nLOCAL VS API: 4 LINES")
    print("1. Choose a local model for privacy, offline use, predictable cost and data control.")
    print("2. Choose an embedding API for managed infrastructure and simpler scaling.")
    print("3. Compare retrieval quality on your own evaluation queries, not only dimensions or speed.")
    print("4. Include network latency, API cost, rate limits and model-version stability in the decision.")

if __name__ == "__main__":
    main()
