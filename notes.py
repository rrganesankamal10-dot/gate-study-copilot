import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Common "filler" words we don't want to highlight (too generic to be useful)
STOPWORDS = {
    "the", "is", "a", "an", "and", "or", "of", "to", "in", "on", "for",
    "with", "as", "by", "at", "from", "this", "that", "it", "be", "are",
    "was", "were", "which", "can", "will", "has", "have", "but", "not",
    "if", "then", "than", "so", "such", "into", "their", "its", "we",
    "you", "your", "i", "he", "she", "they", "them", "his", "her"
}


def clean_chunk_text(text):
    """
    Cleans up raw PDF text a bit:
    - removes extra line breaks
    - removes stray numbers/symbols that PDFs sometimes leave behind
    """
    text = text.replace("\n", " ")          # join broken lines into one
    text = re.sub(r"\s+", " ", text)        # collapse multiple spaces into one
    text = text.strip()
    return text


def extract_keywords_tfidf(texts, max_keywords=5):
    """
    Picks out the most 'important' words for EACH chunk, using TF-IDF.

    The idea: a word is an important keyword for a chunk if it appears
    often in that chunk, but is rare across all the other chunks.
    Common words like "the" or "is" appear everywhere, so they score low.
    Technical terms like "qubit" or "Hilbert" appear in just a few chunks,
    so they score high - those are the words we want to highlight.

    Takes a LIST of texts (one per chunk) so it can compare them to each other.
    Returns a list of keyword-lists, one per input text (same order).
    """
    vectorizer = TfidfVectorizer(
        stop_words="english",   # skip common English filler words automatically
        token_pattern=r"[A-Za-z][A-Za-z\-]{2,}"  # only words with 3+ letters
    )

    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    all_keywords = []
    for row in range(tfidf_matrix.shape[0]):
        row_data = tfidf_matrix[row].toarray()[0]  # scores for this chunk

        # pair up each word with its score, then sort by score (highest first)
        scored_words = list(zip(feature_names, row_data))
        scored_words.sort(key=lambda pair: pair[1], reverse=True)

        # keep only words with a real positive score, skip our own stopwords too
        top_words = [
            word for word, score in scored_words
            if score > 0 and word.lower() not in STOPWORDS
        ][:max_keywords]

        all_keywords.append(top_words)

    return all_keywords


def format_as_notes(chunks):
    """
    Takes search result chunks and turns them into clean note points.
    Each note is a dictionary with:
      - 'point': the cleaned-up text
      - 'page': which page it came from
      - 'keywords': important words to highlight later in the web app
    """
    clean_texts = [clean_chunk_text(chunk["text"]) for chunk in chunks]
    keyword_lists = extract_keywords_tfidf(clean_texts)

    notes = []
    for chunk, clean_text, keywords in zip(chunks, clean_texts, keyword_lists):
        notes.append({
            "point": clean_text,
            "page": chunk["page"],
            "keywords": keywords
        })

    return notes


def summarize_document(all_chunks, num_points=12):
    """
    Picks the most representative sentences from across the WHOLE document,
    to give a quick overview - like a study summary, not a single search answer.

    How it picks:
    - Spreads picks evenly across the document (so early AND late pages are covered,
      not just the first few pages).
    - Within each section, picks the chunk with the strongest TF-IDF keyword signal
      (a proxy for "this chunk says something distinctive/important").
    """
    if not all_chunks:
        return []

    clean_texts = [clean_chunk_text(c["text"]) for c in all_chunks]

    # Skip very short/empty chunks (headers, page numbers, etc.) - not useful as notes
    valid_indices = [i for i, t in enumerate(clean_texts) if len(t) > 80]
    if not valid_indices:
        return []

    valid_texts = [clean_texts[i] for i in valid_indices]
    keyword_lists = extract_keywords_tfidf(valid_texts)

    # Score each valid chunk by how many strong keywords it has (simple importance proxy)
    scored = list(zip(valid_indices, keyword_lists))

    # Split the document into num_points equal sections, pick the best chunk from each
    section_size = max(1, len(scored) // num_points)
    summary_notes = []

    for section_start in range(0, len(scored), section_size):
        section = scored[section_start:section_start + section_size]
        if not section:
            continue
        # pick the chunk in this section with the most keywords found (richest content)
        best_idx, best_keywords = max(section, key=lambda pair: len(pair[1]))

        summary_notes.append({
            "point": clean_texts[best_idx],
            "page": all_chunks[best_idx]["page"],
            "keywords": best_keywords
        })

        if len(summary_notes) >= num_points:
            break

    return summary_notes


# Quick test - only runs if you run this file directly
if __name__ == "__main__":
    from ingest import read_pdf, chunk_text
    from vector_store import embed_chunks, search

    print("Reading and chunking PDF...")
    pages = read_pdf("data/sample.pdf")
    chunks = chunk_text(pages)

    print("Embedding chunks...")
    chunks = embed_chunks(chunks)

    test_question = "What is a qubit?"
    print(f"\nSearching for: '{test_question}'")
    results = search(test_question, chunks, top_k=3)

    notes = format_as_notes(results)

    print("\n--- Generated Notes ---")
    for note in notes:
        print(f"\n• {note['point']}")
        print(f"  (Page {note['page']}) | Keywords: {', '.join(note['keywords'])}")