# Import environment loader
from dotenv import load_dotenv

# Access environment variables
import os

# Import Gemini embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Import Chroma vector database
from langchain_chroma import Chroma

# Import loader functions from loader.py
from loader import load_pdf, split_text


# Load environment variables
load_dotenv()

# Read Gemini API key
api_key = os.getenv("GEMINI_API_KEY")


# Function to create vector database
def create_vector_store():

    # Load PDF text
    text = load_pdf("data/sample.pdf")

    # Split into chunks
    chunks = split_text(text)

    print(f"\nTotal Chunks: {len(chunks)}")

    # Create embedding model
    embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=api_key
)
    # Create ChromaDB vector store
    vectorstore = Chroma.from_texts(

        # Text chunks
        texts=chunks,

        # Embedding model
        embedding=embeddings,

        # Database storage folder
        persist_directory="chroma_db"
    )

    print("\nVector database created successfully!")

    return vectorstore


# Function to test semantic search
def test_search(vectorstore):

    # User query
    query = "What was the internship duration?"

    print(f"\nQuery: {query}")

    # Perform similarity search
    results = vectorstore.similarity_search(query, k=3)

    # Print results
    for i, result in enumerate(results):

        print(f"\n--- RESULT {i+1} ---\n")

        print(result.page_content)


# Main execution block
if __name__ == "__main__":

    # Create vector database
    vectorstore = create_vector_store()

    # Test semantic retrieval
    test_search(vectorstore)