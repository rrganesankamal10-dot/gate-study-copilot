import streamlit as st
import re
from fpdf import FPDF
from ingest import read_pdf, chunk_text
from vector_store import embed_chunks, search
from notes import format_as_notes, summarize_document

# ----- Basic page setup -----
st.set_page_config(
    page_title="Kamal Study Copilot",
    page_icon="📘",
    layout="wide"
)

# ----- Light custom styling for note "cards" -----
st.markdown("""
<style>
.note-card {
    background-color: #1A1D29;
    border: 1px solid #2A2E3F;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 14px;
}
.note-page {
    color: #9A9FB0;
    font-size: 0.85em;
    margin-top: 6px;
}
</style>
""", unsafe_allow_html=True)

# ----- Homescreen header -----
st.title("Kamal Study Copilot")
st.caption("Ask anything from your notes. Get answers with page citations.")
st.divider()

# ----- Session state setup (remembers things between interactions) -----
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None
if "current_notes" not in st.session_state:
    st.session_state.current_notes = None


def highlight_keywords(text, keywords):
    """Wraps each keyword in colored, bold HTML for visual highlighting."""
    highlighted = text
    for keyword in sorted(keywords, key=len, reverse=True):
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        highlighted = pattern.sub(
            f"<span style='color:#8C84FF; font-weight:600;'>{keyword}</span>",
            highlighted
        )
    return highlighted


def render_notes(notes):
    """Displays a list of notes as styled cards with highlighted keywords."""
    for note in notes:
        highlighted_text = highlight_keywords(note["point"], note["keywords"])
        st.markdown(
            f"""<div class="note-card">
                {highlighted_text}
                <div class="note-page">Page {note['page']}</div>
            </div>""",
            unsafe_allow_html=True
        )


def notes_to_pdf_bytes(notes, title="Kamal - Notes"):
    """Builds a simple downloadable PDF from a list of notes."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, title, ln=True)
    pdf.ln(4)

    pdf.set_font("Helvetica", "", 11)
    for note in notes:
        # encode/decode handles characters fpdf's default font can't display
        clean_point = note["point"].encode("latin-1", "replace").decode("latin-1")
        pdf.multi_cell(0, 7, f"- {clean_point}")
        pdf.set_font("Helvetica", "I", 9)
        pdf.cell(0, 6, f"Page {note['page']}", ln=True)
        pdf.set_font("Helvetica", "", 11)
        pdf.ln(2)

    return bytes(pdf.output(dest="S"))


# ----- Sidebar: PDF upload -----
with st.sidebar:
    st.header("Your Document")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file is not None:
        if st.session_state.pdf_name != uploaded_file.name:
            with st.spinner("Reading and understanding your PDF... (first time only)"):
                temp_path = f"data/{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                pages = read_pdf(temp_path)
                chunks = chunk_text(pages)
                chunks = embed_chunks(chunks)

                st.session_state.chunks = chunks
                st.session_state.pdf_name = uploaded_file.name
                st.session_state.current_notes = None

        st.success(f"Loaded: {st.session_state.pdf_name}")
        st.caption(f"{len(st.session_state.chunks)} sections indexed")


# ----- Main area -----
if st.session_state.chunks is None:
    st.info("Upload a PDF from the sidebar to get started.")
else:
    tab_ask, tab_summarize = st.tabs(["Ask a Question", "Summarize"])

    with tab_ask:
        question = st.text_input("Ask a question about your PDF")
        if question:
            with st.spinner("Searching your notes..."):
                results = search(question, st.session_state.chunks, top_k=3)
                notes = format_as_notes(results)
                st.session_state.current_notes = notes

            st.subheader("Notes")
            render_notes(notes)

    with tab_summarize:
        st.write("Get a quick overview of the whole document, organized by page.")
        if st.button("Generate Summary"):
            with st.spinner("Scanning the document for key points..."):
                summary_notes = summarize_document(st.session_state.chunks)
                st.session_state.current_notes = summary_notes

        if st.session_state.current_notes:
            st.subheader("Summary Notes")
            render_notes(st.session_state.current_notes)

    # ----- Download / Print (only show if we have notes to act on) -----
    if st.session_state.current_notes:
        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            pdf_bytes = notes_to_pdf_bytes(st.session_state.current_notes)
            st.download_button(
                label="Download as PDF",
                data=pdf_bytes,
                file_name="kamal_notes.pdf",
                mime="application/pdf"
            )

        with col2:
            if st.button("Print"):
                st.components.v1.html(
                    "<script>window.print()</script>", height=0
                )