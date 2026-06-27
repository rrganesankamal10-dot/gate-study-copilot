from sentence_transformers import SentenceTransformer
import numpy as np

# Load the free local AI model that converts text into "fingerprints" (embeddings).
# This downloads once (~80MB) and then runs fully offline after that.
print("Loading AI model... (first time only, this downloads the model)")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded.")


def embed_chunks(chunks):
    """
    Takes our list of chunks (from ingest.py) and converts each chunk's
    text into an embedding (a list of numbers representing its meaning).

    Returns the same chunks, but each one now also has an 'embedding' field.
    """
    texts = [chunk["text"] for chunk in chunks]  # pull out just the text parts

    print(f"Creating embeddings for {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True)

    # attach each embedding back to its original chunk
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding

    return chunks


def search(query, chunks, top_k=3):
    """
    Given a question (query), find the top_k most relevant chunks.

    How it works:
    1. Convert the query into an embedding (same way we did for chunks).
    2. Compare it to every chunk's embedding using "cosine similarity"
       (a math formula that measures how close two embeddings are).
    3. Return the chunks with the highest similarity scores.
    """
    query_embedding = model.encode([query])[0]

    scored_chunks = []
    for chunk in chunks:
        similarity = cosine_similarity(query_embedding, chunk["embedding"])
        scored_chunks.append((similarity, chunk))

    scored_chunks.sort(key=lambda x: x[0], reverse=True)

    top_chunks = [chunk for score, chunk in scored_chunks[:top_k]]
    return top_chunks


def cosine_similarity(vec_a, vec_b):
    """
    A standard formula to measure how 'close' two embeddings are.
    Returns a number between -1 and 1 (closer to 1 = more similar).
    """
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    return dot_product / (norm_a * norm_b)


# Quick test - only runs if you run this file directly
if __name__ == "__main__":
    from ingest import read_pdf, chunk_text

    print("Reading and chunking PDF...")
    pages = read_pdf("data/sample.pdf")
    chunks = chunk_text(pages)

    print("Embedding chunks (this may take a minute for a big PDF)...")
    chunks = embed_chunks(chunks)

    test_question = "What is a qubit?"
    print(f"\nSearching for: '{test_question}'")
    results = search(test_question, chunks, top_k=3)

    print("\nTop matching chunks:")
    for i, chunk in enumerate(results, 1):
        print(f"\n--- Result {i} (Page {chunk['page']}) ---")
        print(chunk["text"][:200])  # print first 200 characters