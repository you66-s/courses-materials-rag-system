import sys
import os, uuid
import tempfile

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

import streamlit as st
from retriever import Retriever
from generator import Generator
from embeddings_model import EmbeddingsModel
from vectorDB import VectorDataBase
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Courses Materials RAG Assistant",
    page_icon="📘",
    layout="wide"
)

# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------
custom_css = """
<style>
:root {
    --user-bg: #DCF8C6;
    --bot-bg: #E8E8E8;
    --border-color: #CCCCCC;
}

.chat-message {
    padding: 12px;
    border-radius: 12px;
    margin: 8px 0;
    max-width: 85%;
    line-height: 1.5;
}

.user-bubble {
    background-color: var(--user-bg);
    margin-left: auto;
    border: 1px solid var(--border-color);
}

.bot-bubble {
    background-color: var(--bot-bg);
    margin-right: auto;
    border: 1px solid var(--border-color);
}

.source-card {
    border: 1px solid #999;
    border-radius: 10px;
    padding: 10px;
    margin: 6px 0;
    background: white;
    box-shadow: 0px 2px 6px rgba(0,0,0,0.08);
}

.source-title {
    font-weight: bold;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------
st.title("📘 Courses Materials RAG System")
st.write("Ask questions about your uploaded course materials.")

# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "collection_name" not in st.session_state:
    st.session_state.collection_name = f"session_{uuid.uuid4().hex}"
if "indexed" not in st.session_state:
    st.session_state.indexed = False
# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.header("📂 Document Upload")

uploaded_files = st.sidebar.file_uploader(
    "Upload course materials (PDF, TXT)",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

# Store uploaded files (avoid duplicates)
if uploaded_files:
    for file in uploaded_files:
        if file.name not in [f["name"] for f in st.session_state.uploaded_files]:
            st.session_state.uploaded_files.append({
                "name": file.name,
                "type": file.type,
                "size": file.size,
                "file": file
            })


# ---------------------------------------------------------
# SIDEBAR SETTINGS
# ---------------------------------------------------------
st.sidebar.header("🔧 Retrieval Settings")

top_k = st.sidebar.slider("Top-K retrieved chunks", 1, 10, 5)

module_filter = st.sidebar.text_input(
    "Filter by module (optional)",
    placeholder="Probabilité, IA, Réseaux..."
)

# ---------------------------------------------------------
# INDEX FILES INTO RETRIEVER
# ---------------------------------------------------------
def index_uploaded_files():
    embedding_model = EmbeddingsModel()
    vector_db = VectorDataBase(collection_name=st.session_state.collection_name)
    text_splitter = RecursiveCharacterTextSplitter( chunk_size=1000, chunk_overlap=200, length_function=len, separators=["\n\n", "\n", " ", ""] )
    print("Starting indexing")
    for f in st.session_state.uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            f["file"].seek(0)
            tmp.write(f["file"].read())
            tmp_path = tmp.name
        try:
            data_loader = PyMuPDFLoader(file_path=tmp_path, mode='page')
            documents = data_loader.load()
            chunks = text_splitter.split_documents(documents=documents)
            for chunk in chunks:
                embedding = embedding_model.embed_texts([chunk.page_content])[0]
                vector_db.add_document(
                    id=str(uuid.uuid4()),
                    document=chunk.page_content,
                    metadata=chunk.metadata,
                    embedding=embedding
                )
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            

# Uncomment once retriever ingestion is ready
if st.session_state.uploaded_files and not st.session_state.indexed:
    index_uploaded_files()
    st.session_state.indexed = True

@st.cache_resource
def load_retriever():
    return Retriever(collection_name=st.session_state.collection_name)
retriever_inst = load_retriever()





# ---------------------------------------------------------
# USER QUERY INPUT
# ---------------------------------------------------------
query = st.text_input(
    "Your question",
    placeholder="Example: C'est quoi la probabilité ?"
)

submit = st.button("Run", use_container_width=True)



# ---------------------------------------------------------
# PROCESS QUERY
if submit and query.strip():
    print("start query processing...")
    with st.spinner("🔍 Retrieving context and generating answer..."):
        try:
            generator_inst = Generator(query=query)
            print("generator initialized...")
            response = generator_inst.generate(
                retreiver=retriever_inst,
                top_k=top_k
            )
            print("response retrieved...")
            st.session_state.messages.append({
                "query": query,
                "response": response
            })
            print("retrieved done...")
        except Exception as e:
            print(f"Error: {e}")

# ---------------------------------------------------------
# DISPLAY CHAT HISTORY
# ---------------------------------------------------------
for msg in st.session_state.messages:

    # USER MESSAGE
    st.markdown(
        f"""
        <div class="chat-message user-bubble">
            <strong>You:</strong><br>
            {msg['query']}
        </div>
        """,
        unsafe_allow_html=True
    )

    # ASSISTANT MESSAGE
    st.markdown(
        f"""
        <div class="chat-message bot-bubble">
            <strong>Assistant:</strong><br>
            {msg['response']}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")
