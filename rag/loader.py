# Import PDF reader
from PyPDF2 import PdfReader

# Import LangChain text splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Function to load PDF and extract text
def load_pdf(pdf_path):

    # Create PDF reader object
    reader = PdfReader(pdf_path)

    # Store all text here
    full_text = ""

    # Loop through all PDF pages
    for page in reader.pages:

        # Extract text from current page
        text = page.extract_text()

        # Add page text to full_text
        if text:
            full_text += text

    return full_text


# Function to split text into chunks
def split_text(text):

    # Create text splitter object
    splitter = RecursiveCharacterTextSplitter(

        # Maximum characters per chunk
        chunk_size=500,

        # Overlapping characters between chunks
        chunk_overlap=50
    )

    # Split text into chunks
    chunks = splitter.split_text(text)

    return chunks


# Main test block
if __name__ == "__main__":

    # PDF file path
    pdf_path = "data/sample.pdf"

    # Load PDF text
    text = load_pdf(pdf_path)

    # Print first 500 characters
    print("\nFIRST 500 CHARACTERS:\n")
    print(text[:500])

    # Split into chunks
    chunks = split_text(text)

    # Print number of chunks
    print(f"\nTotal Chunks Created: {len(chunks)}\n")

    # Print first 3 chunks
    for i, chunk in enumerate(chunks[:3]):

        print(f"\n--- CHUNK {i+1} ---\n")

        print(chunk)