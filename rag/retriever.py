# Import environment loader
from dotenv import load_dotenv

# Access environment variables
import os

# Import Gemini client
from google import genai

# Import Gemini embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Import Chroma vector database
from langchain_chroma import Chroma


# Load environment variables
load_dotenv()

# Read Gemini API key
api_key = os.getenv("GEMINI_API_KEY")

# Create Gemini client
client = genai.Client(api_key=api_key)


# Load embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=api_key
)


# Load existing ChromaDB
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)


# Function to answer questions
def ask_question(question):

    print(f"\nQuestion: {question}")

    # Retrieve top 3 relevant chunks
    results = vectorstore.similarity_search(question, k=3)

    print("\n========== RETRIEVED CHUNKS ==========\n")

    # Store retrieved text here
    context = ""

    # Print retrieved chunks
    for i, result in enumerate(results):

        print(f"\n--- CHUNK {i+1} ---\n")

        print(result.page_content)

        # Add chunk to context
        context += result.page_content + "\n\n"

    # Build prompt
    prompt = f"""
You are a helpful AI assistant.

Answer the question ONLY using the provided context.

If the answer is not present in the context, say:
"I could not find the answer in the PDF."

Context:
{context}

Question:
{question}

Answer:
"""

    print("\n========== FINAL PROMPT ==========\n")

    print(prompt)

    # Send prompt to Gemini
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    # Print final answer
    print("\n========== FINAL ANSWER ==========\n")

    print(response.text)


# Main execution
if __name__ == "__main__":

    # Example question
    question = "What company hosted the internship?"
    # Run RAG pipeline
    ask_question(question)