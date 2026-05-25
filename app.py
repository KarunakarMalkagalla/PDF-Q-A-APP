# Import Streamlit
import streamlit as st

# Load environment variables
from dotenv import load_dotenv

# Access environment variables
import os

# Gemini SDK
from google import genai

# Gemini embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# ChromaDB
from langchain_chroma import Chroma

# PDF reader
from PyPDF2 import PdfReader

# Text splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(
    page_title="PDF Q&A App",
    page_icon="📘",
    layout="wide"
)


# ==============================
# CUSTOM CSS
# ==============================

st.markdown("""
<style>

.main {
    padding: 2rem;
}

.stChatMessage {
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 1rem;
}

.source-box {
    background-color: #f0f2f6;
    color: black;
    padding: 15px;
    border-radius: 10px;
    margin-top: 10px;
    font-size: 15px;
}

</style>
""", unsafe_allow_html=True)


# ==============================
# LOAD API KEY
# ==============================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

# Create Gemini client
client = genai.Client(api_key=api_key)


# ==============================
# SIDEBAR
# ==============================

st.sidebar.title("📄 PDF Q&A App")

st.sidebar.write("Upload a PDF and ask questions.")

# Clear chat button
if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []


# ==============================
# SESSION STATE
# ==============================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ==============================
# PDF UPLOADER
# ==============================

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type="pdf"
)


# ==============================
# PROCESS PDF
# ==============================

if uploaded_file:

    # Save uploaded PDF
    pdf_path = f"data/{uploaded_file.name}"

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("PDF uploaded successfully!")

    # Read PDF
    reader = PdfReader(pdf_path)

    # Extract text
    full_text = ""

    for page in reader.pages:

        text = page.extract_text()

        if text:
            full_text += text

    # Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_text(full_text)

    # Sidebar info
    st.sidebar.write(f"📚 Chunks: {len(chunks)}")

    # Create embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=api_key
    )

    # ==============================
    # CREATE OR LOAD VECTOR DB
    # ==============================

    if not os.path.exists("chroma_db"):

        # Create vector DB only once
        vectorstore = Chroma.from_texts(
            texts=chunks,
            embedding=embeddings,
            persist_directory="chroma_db"
        )

        st.success("Vector database created!")

    else:

        # Load existing DB
        vectorstore = Chroma(
            persist_directory="chroma_db",
            embedding_function=embeddings
        )

        st.success("Loaded existing vector database!")

    # ==============================
    # CHAT INPUT
    # ==============================

    user_question = st.chat_input(
        "Ask a question about the PDF..."
    )

    if user_question:

        # Store user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_question
        })

        # Retrieve relevant chunks
        results = vectorstore.similarity_search(
            user_question,
            k=3
        )

        # Build context
        context = ""

        for result in results:
            context += result.page_content + "\n\n"

        # Build prompt
        prompt = f"""
You are a helpful AI assistant.

Answer ONLY from the provided PDF context.

If the answer is not found, say:
"I could not find the answer in the PDF."

Context:
{context}

Question:
{user_question}

Answer:
"""

        # Generate answer
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        answer = response.text

        # Store assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": results
        })


# ==============================
# DISPLAY CHAT
# ==============================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.write(message["content"])

        # Show source chunks
        if message["role"] == "assistant":

            if "sources" in message:

                st.markdown("### 📚 Source Chunks")

                for i, source in enumerate(message["sources"]):

                    st.markdown(f"""
<div class="source-box">

<b>Chunk {i+1}</b><br><br>

{source.page_content}

</div>
""", unsafe_allow_html=True)