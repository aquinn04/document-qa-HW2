import streamlit as st
from openai import OpenAI
import os
import sys

__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
from chromadb.utils import embedding_functions
from bs4 import BeautifulSoup

st.title("HW 4 - iSchool Student Organizations Chatbot")

# pull the key from Streamlit secrets.
if "openai_api_key" not in st.secrets:
    st.error("Please add your openai_api_key to secrets.toml")
    st.stop()

client = OpenAI(api_key=st.secrets["openai_api_key"])
MODEL = "gpt-4o-mini"

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=st.secrets["openai_api_key"],
    model_name="text-embedding-3-small",
)

# create chromaDB client
Chroma_client = chromadb.PersistentClient(path='./ChromaDB_for_HW4')


def chunk_html_file(filepath):
    """
    Chunking method: fixed 2-way split by character count.
    Each HTML file is parsed down to plain text (stripping tags),
    then split into two roughly equal halves by character length.
    This is a simple, deterministic method: it guarantees no single
    chunk is too large for the embedding model, and gives the
    retriever two separate "shots" at matching a user's query against
    different portions of the same page (e.g., an org's description
    vs. its meeting times/contact info), without needing more complex
    sentence- or paragraph-boundary detection.
    """
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    text = soup.get_text(separator=" ", strip=True)

    midpoint = len(text) // 2
    chunk1 = text[:midpoint]
    chunk2 = text[midpoint:]
    return [chunk1, chunk2]


def create_hw4_collection():
    if 'HW4_VectorDB' not in st.session_state:
        collection = Chroma_client.get_or_create_collection(
            "HW4Collection", embedding_function=openai_ef
        )
        html_folder = "./html_docs"

        if os.path.isdir(html_folder):
            for filename in os.listdir(html_folder):
                filepath = os.path.join(html_folder, filename)
                if not os.path.isfile(filepath) or not filename.lower().endswith(".html"):
                    continue

                chunks = chunk_html_file(filepath)
                for i, chunk_text in enumerate(chunks):
                    if chunk_text.strip():
                        collection.upsert(
                            documents=[chunk_text],
                            metadatas=[{"filename": filename, "chunk": i}],
                            ids=[f"{filename}_chunk{i}"],
                        )

        st.session_state.HW4_VectorDB = collection

    return st.session_state.HW4_VectorDB


collection = create_hw4_collection()

# AI Prompts
SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are a helpful chatbot that answers questions about iSchool "
        "student organizations. Be clear and friendly. "
        "If you are given document excerpts to help answer the question, "
        "you MUST clearly say so at the start of your answer, e.g., "
        "'Based on the student organization pages, ...'. If no relevant "
        "documents were retrieved, answer from your general knowledge instead."
    ),
}

if "hw4_messages" not in st.session_state:
    st.session_state.hw4_messages = [SYSTEM_PROMPT]

# chat history
for msg in st.session_state.hw4_messages:
    if msg["role"] == "system":
        continue
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


def get_last_five_interactions_buffer(all_messages):
    """
    Memory buffer: keep the system prompt always, plus the last 5
    user/assistant interactions (an 'interaction' = one user message +
    one assistant reply, so up to 10 messages total besides system).
    """
    system_msgs = [m for m in all_messages if m["role"] == "system"]
    convo = [m for m in all_messages if m["role"] != "system"]
    trimmed = convo[-10:]  # last 5 user + 5 assistant messages (at most)
    return system_msgs + trimmed


# chat inputs
if prompt := st.chat_input("Ask me anything about student organizations..."):
    # 1. Save + show the user's message
    st.session_state.hw4_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. RAG retrieval: search the vector DB with the user's question
    rag_results = st.session_state.HW4_VectorDB.query(
        query_texts=[prompt],
        n_results=3
    )
    retrieved_docs = rag_results["documents"][0]
    retrieved_ids = rag_results["ids"][0]

    context_text = "\n\n".join(
        f"[{doc_id}]\n{doc[:1000]}"
        for doc_id, doc in zip(retrieved_ids, retrieved_docs)
    )

    rag_context_msg = {
        "role": "system",
        "content": (
            "Here are relevant excerpts from student organization pages "
            "that may help answer the user's question:\n\n" + context_text
        ),
    }

    # 3. Build the buffered message list, with RAG context injected
    messages_to_send = get_last_five_interactions_buffer(
        st.session_state.hw4_messages
    ) + [rag_context_msg]

    # 4. Stream the response
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=MODEL,
            messages=messages_to_send,
            stream=True,
        )
        response = st.write_stream(stream)

    # 5. Save the assistant's full response into history
    st.session_state.hw4_messages.append({"role": "assistant", "content": response})