# Kamal Study Copilot

Ask questions about any PDF and get answers grounded in the actual document, with page citations — no API key, no cost, runs locally.

## What it does

Upload any PDF (lecture notes, research papers, textbooks). The app reads it, understands its content using AI, and lets you:
- **Ask a question** → get the most relevant passages from the document, with page numbers
- **Summarize** → get a quick overview of the whole document, spread across all sections
- **Download or print** the generated notes

## Why I built this

I'm preparing for GATE ECE and campus placements, and I kept needing to dig through long PDFs (syllabus documents, reference books, papers) to find specific answers quickly. Most "chat with your PDF" tools need a paid API key. I wanted something that works completely offline and free, so I built it myself.

## How it works (architecture)

This is a RAG (Retrieval-Augmented Generation) pipeline — without the "Generation" part needing a paid LLM. Here's the flow:

1. **PDF → Text** (`ingest.py`)
   PyMuPDF reads the PDF page by page, extracting raw text.

2. **Text → Chunks** (`ingest.py`)
   Each page's text is split into ~500-character chunks. Each chunk remembers which page it came from — this is what makes page citations possible later.

3. **Chunks → Embeddings** (`vector_store.py`)
   Each chunk is converted into a 384-number "embedding" using a free local AI model (`sentence-transformers/all-MiniLM-L6-v2`). An embedding is a numeric representation of meaning — chunks with similar meaning get similar numbers, even if they use different words.

4. **Question → Search** (`vector_store.py`)
   When you ask a question, it's converted into an embedding the same way. Then **cosine similarity** (a standard math formula for comparing vectors) finds the chunks whose embeddings are closest to the question's embedding — i.e., the most relevant content.

5. **Chunks → Readable Notes** (`notes.py`)
   Raw PDF text is messy (broken lines, stray symbols). This step cleans it up and extracts keywords using **TF-IDF** (Term Frequency–Inverse Document Frequency) — a technique that finds words that are distinctive to a specific chunk compared to the rest of the document. These keywords get highlighted in the UI.

6. **Summarization** (`notes.py`)
   Rather than calling a paid AI to generate new sentences, the summarizer splits the document into even sections and picks the most keyword-rich chunk from each — this is **extractive summarization**, not generative. It's honest about being a smart selection of existing content, not a rewrite.

7. **Web Interface** (`app.py`)
   Built with Streamlit. Handles file upload, session state (so it doesn't reprocess the PDF on every click), the Q&A and Summarize tabs, keyword highlighting, and PDF export (via `fpdf2`).

## Tech stack

- **Python** — core language
- **PyMuPDF (fitz)** — PDF text extraction
- **sentence-transformers** — free local embedding model (all-MiniLM-L6-v2)
- **scikit-learn** — TF-IDF keyword extraction
- **Streamlit** — web interface
- **fpdf2** — PDF generation for notes export

No paid APIs. No API keys. Everything after the one-time model download runs fully offline.

## Known limitations (and what I'd improve next)

Being upfront about this matters more than pretending it's perfect:

- **Fixed-size chunking** (500 characters) sometimes splits sentences awkwardly. A better version would chunk by sentence or paragraph boundaries.
- **No similarity threshold** — the search always returns exactly 3 results, even if none are a strong match. A real improvement: only return chunks above a minimum similarity score, and say "no strong match found" otherwise.
- **Extractive, not generative, summarization** — it selects existing sentences rather than writing new ones. This was a deliberate tradeoff for staying free/local, not an oversight.
- **TF-IDF keyword extraction** sometimes picks distinctive-but-not-conceptually-central words (e.g., "pumps" in a chunk about refrigerator analogies). A more advanced version could use a different keyword extraction model.
- **Processing time scales with document size** — larger PDFs take longer to embed since everything runs on a local CPU, not a cloud GPU.

## How to explain this in an interview

If asked **"what is RAG and why not just paste the whole PDF into a prompt?"**
→ Large documents don't fit in a single prompt's context window, and even when they do, models attend to relevant information better when given a focused, retrieved excerpt rather than everything at once. RAG retrieves only the relevant pieces first.

If asked **"why 500-character chunks?"**
→ It's a simple, fixed-size starting point that balances having enough context per chunk against keeping search precise. A production system would chunk by semantic boundaries (paragraphs/sentences) instead.

If asked **"what's an embedding, in your own words?"**
→ A list of numbers that represents the *meaning* of a piece of text, generated by a neural network trained on huge amounts of text. Texts with similar meaning end up with numerically similar embeddings.

If asked **"what happens if the answer isn't in the PDF?"**
→ Currently, it still returns the 3 closest chunks regardless of how weak the match is — a real limitation. The fix would be a similarity score threshold.

If asked **"why didn't you use a paid AI model to generate answers?"**
→ Deliberate choice to keep the tool fully free and usable without any signup or API key, important for a tool meant for personal study use. The tradeoff is that answers are retrieved excerpts, not freshly written paragraphs.

If asked **"how would you scale this to a 1000-page textbook?"**
→ Store embeddings in a proper vector database (like FAISS or Chroma) instead of recalculating them every time, and consider batching/caching the embedding step so it only happens once per document, persisted to disk.
