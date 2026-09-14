import streamlit as st
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import anthropic

st.title("HW3 - Streaming Chatbot that Discusses a URL")

st.write(
    "This chatbot answers questions about up to two web pages you give it. "
    "Paste one or two URLs in the sidebar, click 'Load URLs', pick which LLM "
    "to chat with, then ask questions below. The page content is stored as a "
    "system message that is never removed, so the chatbot always has access "
    "to it. On top of that, the chatbot remembers the last 3 exchanges "
    "(6 messages) of your conversation, so older turns get dropped once the "
    "conversation runs long."
)

# sidebar controls
url1 = st.sidebar.text_input("URL 1")
url2 = st.sidebar.text_input("URL 2 (optional)")

llm_choice = st.sidebar.selectbox(
    "Choose LLM",
    ["OpenAI (GPT-5)", "Claude (Opus 5)"],
)


# reads a webpage and returns its visible text content
def read_url_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # raise an exception for HTTP errors
        soup = BeautifulSoup(response.content, "html.parser")
        return soup.get_text()
    except requests.RequestException as e:
        st.error(f"Error reading {url}: {e}")
        return None


if "messages" not in st.session_state:
    st.session_state.messages = []

if st.sidebar.button("Load URLs"):
    contents = []

    if url1:
        text1 = read_url_content(url1)
        if text1:
            contents.append(f"Content from {url1}:\n{text1}")

    if url2:
        text2 = read_url_content(url2)
        if text2:
            contents.append(f"Content from {url2}:\n{text2}")

    if contents:
        combined = "\n\n".join(contents)
        system_prompt = {
            "role": "system",
            "content": (
                "You are a helpful assistant. Use the following webpage "
                "content to answer the user's questions. If the answer isn't "
                "in the content, say so.\n\n" + combined
            ),
        }
        # starting a fresh conversation with the new system prompt
        st.session_state.messages = [system_prompt]
        st.success("URLs loaded. You can start chatting below.")
    else:
        st.warning("Please enter at least one URL.")

# show the chat history, skipping the system message
for msg in st.session_state.messages:
    if msg["role"] == "system":
        continue
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# keeps the system message, plus only the last 6 messages (3 exchanges)
def get_buffered_messages():
    system_msgs = [m for m in st.session_state.messages if m["role"] == "system"]
    convo = [m for m in st.session_state.messages if m["role"] != "system"]
    return system_msgs + convo[-6:]


if prompt := st.chat_input("Ask a question about the page(s)"):
    has_system_prompt = any(
        m["role"] == "system" for m in st.session_state.messages
    )

    if not has_system_prompt:
        st.warning("Please load at least one URL in the sidebar first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        messages_to_send = get_buffered_messages()
        response = ""

        with st.chat_message("assistant"):
            if llm_choice == "OpenAI (GPT-5)":
                openai_api_key = st.secrets["openai_api_key"]
                client = OpenAI(api_key=openai_api_key)

                try:
                    stream = client.chat.completions.create(
                        model="gpt-5",
                        messages=messages_to_send,
                        stream=True,
                    )
                    response = st.write_stream(stream)
                except Exception as e:
                    st.error(f"OpenAI request failed. Check your API key. ({e})")

            else:
                anthropic_api_key = st.secrets["anthropic_api_key"]
                client = anthropic.Anthropic(api_key=anthropic_api_key)

                # anthropic keeps the system prompt separate from the messages list
                system_text = messages_to_send[0]["content"]
                claude_messages = [
                    m for m in messages_to_send if m["role"] != "system"
                ]

                try:
                    with client.messages.stream(
                        model="claude-opus-5",
                        max_tokens=1024,
                        system=system_text,
                        messages=claude_messages,
                    ) as stream:
                        response = st.write_stream(stream.text_stream)
                except Exception as e:
                    st.error(f"Claude request failed. Check your API key. ({e})")

        if response:
            st.session_state.messages.append(
                {"role": "assistant", "content": response}
            )