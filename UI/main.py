import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import streamlit as st
from backend.retriever import Retriever
from backend.generator import Generator

# TODO: Inverse the chat user unput and asisstant response order
# TODO: Fix the hallucination issue in the generator
# TODO: Change the model
# TODO: enhace the visibility of assistant response UI.



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

/* Chat container */
.chat-message {
    padding: 12px;
    border-radius: 12px;
    margin: 8px 0;
    max-width: 85%;
    line-height: 1.5;
}

/* User bubble */
.user-bubble {
    background-color: var(--user-bg);
    margin-left: auto;
    border: 1px solid var(--border-color);
}

/* Bot bubble */
.bot-bubble {
    background-color: var(--bot-bg);
    margin-right: auto;
    border: 1px solid var(--border-color);
}

/* Source cards */
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
    color: #333;
}

/* Animated run button */
.run-button button {
    background: linear-gradient(90deg, #0066FF, #00CCFF);
    color: white !important;
    border-radius: 8px !important;
    border: none;
    padding: 10px 18px;
    font-size: 16px;
    animation: glow 2s infinite;
}

@keyframes glow {
    0% { box-shadow: 0 0 5px #00A3FF; }
    50% { box-shadow: 0 0 18px #00E0FF; }
    100% { box-shadow: 0 0 5px #00A3FF; }
}

</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)


# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------
st.title("📘 Courses Materials RAG System")
st.write("Ask anything about your course materials. The model retrieves relevant chunks and generates an answer.")


# ---------------------------------------------------------
# INITIALIZE RETRIEVER ONCE
# ---------------------------------------------------------
@st.cache_resource
def load_retriever():
    return Retriever()

retriever = load_retriever()


# ---------------------------------------------------------
# SESSION CHAT HISTORY
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------------------------------------------------
# SIDEBAR: Filters
# ---------------------------------------------------------
st.sidebar.header("🔧 Settings")

top_k = st.sidebar.slider("Top-K retrieved chunks", 1, 10, 5)

module_filter = st.sidebar.text_input(
    "Filter by module name (optional)",
    placeholder="e.g., Probabilité, IA, Réseaux…"
)

st.sidebar.write("These filters influence retrieval and context selection.")


# ---------------------------------------------------------
# USER INPUT
# ---------------------------------------------------------
query = st.text_input("❓ Your question:", placeholder="Example: C'est quoi la probabilité ?")

run_button = st.container()
with run_button:
    submit = st.button("🚀 Run", key="run", use_container_width=True)


# ---------------------------------------------------------
# PROCESS QUERY
# ---------------------------------------------------------
if submit and query.strip():
    with st.spinner("🔍 Retrieving context and generating answer…"):
        try:
            generator = Generator(query=query)
            response = generator.generate(
                retreiver=retriever,
                top_k=top_k
            )

            # Append to chat history
            st.session_state.messages.append({
                "query": query,
                "response": response
            })

        except Exception as e:
            st.error(f"❌ Error: {e}")


# ---------------------------------------------------------
# DISPLAY CHAT HISTORY WITH BUBBLES
# ---------------------------------------------------------
for msg in st.session_state.messages:
    
    # USER bubble
    st.markdown(
        f"""
        <div class="chat-message user-bubble">
            <strong>You:</strong><br>
            {msg['query']}
        </div>
        """,
        unsafe_allow_html=True
    )

    # BOT bubble
    st.markdown(
        f"""
        <div class="chat-message bot-bubble">
            <strong>Assistant:</strong><br>
            {msg['response']}
        </div>
        """,
        unsafe_allow_html=True
    )

    # ---------------------------------------------------------
    # DISPLAY SOURCES AS CARDS
    # ---------------------------------------------------------
    if "sources" in msg["response"]:
        st.markdown("##### 📚 Sources Used")
        for src in msg["response"]["sources"]:
            st.markdown(
                f"""
                <div class="source-card">
                    <div class="source-title">📄 Source</div>
                    <div><strong>Module:</strong> {src.get('module', 'N/A')}</div>
                    <div><strong>Page:</strong> {src.get('page', 'N/A')}</div>
                    <div><strong>Chunk ID:</strong> {src.get('source_id', 'N/A')}</div>
                    <div><strong>Metadata:</strong> {src}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("---")
