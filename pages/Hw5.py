import streamlit as st
from openai import OpenAI
import os
import sys
import json

__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
from chromadb.utils import embedding_functions
from bs4 import BeautifulSoup

st.title("HW 5 - Enhanced iSchool Student Organizations Chatbot")

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

# create chromaDB client (same DB folder/collection as HW4)
Chroma_client = chromadb.PersistentClient(path='./ChromaDB_for_HW4')


def chunk_html_file(filepath):
    """
    Chunking method: fixed 2-way split by character count.
    Each HTML file is parsed down to plain text (stripping tags),
    then split into two roughly equal halves by character length.
    This is a simple, deterministic method: it guarantees no single
    chunk is too large for the embedding model, and gives the
    retriever two separate "shots" at matching a query against
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


def create_hw5_collection():
    if 'HW5_VectorDB' not in st.session_state:
        collection = Chroma_client.get_or_create_collection(
            "HW4Collection", embedding_function=openai_ef
        )
        html_folder = "./html_docs"

        if os.path.isdir(html_folder):
            html_files = []
            for root, _dirs, files in os.walk(html_folder):
                for fn in files:
                    if fn.lower().endswith(".html"):
                        html_files.append((fn, os.path.join(root, fn)))

            for filename, filepath in html_files:
                chunks = chunk_html_file(filepath)
                for i, chunk_text in enumerate(chunks):
                    if chunk_text.strip():
                        collection.upsert(
                            documents=[chunk_text],
                            metadatas=[{"filename": filename, "chunk": i}],
                            ids=[f"{filename}_chunk{i}"],
                        )

        st.session_state.HW5_VectorDB = collection

    return st.session_state.HW5_VectorDB


collection = create_hw5_collection()


# ---------------------------------------------------------------------------
# The tool: the LLM supplies the 'query'; we do the vector search.
# ---------------------------------------------------------------------------
def relevant_club_info(query):
    """Vector search over the ChromaDB collection; returns relevant excerpts as text."""
    results = st.session_state.HW5_VectorDB.query(
        query_texts=[query],
        n_results=3,
    )
    docs = results["documents"][0]
    ids = results["ids"][0]

    if not docs:
        return "No relevant information found."

    return "\n\n".join(
        f"[{doc_id}]\n{doc[:1000]}" for doc_id, doc in zip(ids, docs)
    )


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "relevant_club_info",
            "description": (
                "Search the iSchool student organization pages and return the "
                "excerpts most relevant to the query. Use this whenever the "
                "user asks about a specific club or about student organizations "
                "in general (meetings, contacts, purpose, events, how to join)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "A concise search query capturing what club "
                            "information is needed, e.g. 'Data Science Club meeting times'."
                        ),
                    }
                },
                "required": ["query"],
            },
        },
    }
]

# AI Prompts
SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are a helpful chatbot that answers questions about iSchool "
        "student organizations. Be clear and friendly. "
        "When the user asks about a club or student organizations, call the "
        "relevant_club_info tool with a concise search query before answering. "
        "For greetings or follow-ups you can answer from the conversation "
        "history, you do not need to call the tool. "
        "If you were given document excerpts from the tool, you MUST clearly "
        "say so at the start of your answer, e.g., 'Based on the student "
        "organization pages, ...'. If the excerpts do not contain the answer, "
        "say so honestly rather than guessing; if no documents were retrieved, "
        "you may answer from your general knowledge and say that."
    ),
}

if "hw5_messages" not in st.session_state:
    st.session_state.hw5_messages = [SYSTEM_PROMPT]

# chat history
for msg in st.session_state.hw5_messages:
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


def generate_answer(messages_to_send):
    """
    Pass 1: LLM may call relevant_club_info (it writes the query).
    Pass 2: LLM is invoked again with the tool results and NO tools
            available, so it must answer from what was retrieved.
    Yields streamed text.
    """
    first = client.chat.completions.create(
        model=MODEL,
        messages=messages_to_send,
        tools=TOOLS,
        tool_choice="auto",
    )
    first_msg = first.choices[0].message

    # No tool needed (greeting, follow-up, etc.)
    if not first_msg.tool_calls:
        yield first_msg.content or ""
        return

    assistant_tool_msg = {
        "role": "assistant",
        "content": first_msg.content,
        "tool_calls": [
            {
                "id": c.id,
                "type": "function",
                "function": {"name": c.function.name, "arguments": c.function.arguments},
            }
            for c in first_msg.tool_calls
        ],
    }
    followup_messages = messages_to_send + [assistant_tool_msg]

    for call in first_msg.tool_calls:
        args = json.loads(call.function.arguments or "{}")
        query = args.get("query", "")
        with st.sidebar:
            st.caption(f"🔎 Tool called: relevant_club_info('{query}')")
        followup_messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "content": relevant_club_info(query),
        })

    # Second invocation: results shown to the LLM, no tools offered.
    stream = client.chat.completions.create(
        model=MODEL,
        messages=followup_messages,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


# chat inputs
if prompt := st.chat_input("Ask me anything about student organizations..."):
    # 1. Save + show the user's message
    st.session_state.hw5_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Build the buffered message list (no prompt-based embedding anymore)
    messages_to_send = get_last_five_interactions_buffer(
        st.session_state.hw5_messages
    )

    # 3. Tool-based retrieval + streamed response
    with st.chat_message("assistant"):
        response = st.write_stream(generate_answer(messages_to_send))

    # 4. Save the assistant's full response into history
    st.session_state.hw5_messages.append({"role": "assistant", "content": response})