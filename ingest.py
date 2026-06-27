import fitz  # this is the PyMuPDF library, used to read PDFs

def read_pdf(pdf_path):
    """
    Opens a PDF file and returns its text, page by page.
    Returns a list where each item is one page's text.
    """
    doc = fitz.open(pdf_path)  # open the PDF file
    pages_text = []  # empty list to store text from each page

    for page_number in range(len(doc)):  # loop through every page
        page = doc[page_number]  # get one page
        text = page.get_text()  # extract the text from that page
        pages_text.append(text)  # save it in our list

    doc.close()  # close the file when done
    return pages_text


def chunk_text(pages_text, chunk_size=500):
    """
    Takes the list of page texts and breaks them into smaller chunks.
    Each chunk remembers which page it came from (for citations later).
    chunk_size = roughly how many characters per chunk.
    """
    chunks = []  # list to store all chunks

    for page_number, text in enumerate(pages_text, start=1):
        text = text.strip()  # remove extra blank spaces
        if not text:  # skip empty pages
            continue

        # break this page's text into pieces of chunk_size characters
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            chunks.append({
                "text": chunk,
                "page": page_number
            })

    return chunks


# This part only runs if you run this file directly (for testing)
if __name__ == "__main__":
    test_pdf_path = "data/sample.pdf"  # we'll add a real PDF here soon

    print("Reading PDF...")
    pages = read_pdf(test_pdf_path)
    print(f"Found {len(pages)} pages.")

    print("Breaking into chunks...")
    chunks = chunk_text(pages)
    print(f"Created {len(chunks)} chunks.")

    print("\nFirst chunk preview:")
    print(chunks[0])
    