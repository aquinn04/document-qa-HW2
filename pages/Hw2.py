import streamlit as st
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import anthropic

st.title("HW2 - URL Summarizer with Multiple LLMs")

# url input goes at the top of the page, not the sidebar
url = st.text_input("Enter a URL to summarize")

# sidebar controls
summary_type = st.sidebar.selectbox(
    "Summarize document in",
    ["100 words", "2 connecting paragraphs", "5 bullet points"],
)

language = st.sidebar.selectbox(
    "Output language",
    ["English", "French", "Spanish", "German"],
)

llm_choice = st.sidebar.selectbox(
    "Choose LLM",
    ["OpenAI", "Claude"],
)

use_advanced = st.sidebar.checkbox("Use advanced model")


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


if url:
    document = read_url_content(url)

    if document:
        prompt = (
            f"Here's the content of a webpage: {document}\n\n"
            f"Summarize this content in {summary_type}. "
            f"Respond only in {language}."
        )

        if llm_choice == "OpenAI":
            openai_api_key = st.secrets["openai_api_key"]
            client = OpenAI(api_key=openai_api_key)
            model = "gpt-5" if use_advanced else "gpt-5-mini"

            try:
                stream = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    stream=True,
                )
                st.write_stream(stream)
            except Exception as e:
                st.error(f"OpenAI request failed. Check your API key. ({e})")

        elif llm_choice == "Claude":
            anthropic_api_key = st.secrets["anthropic_api_key"]
            client = anthropic.Anthropic(api_key=anthropic_api_key)
            model = "claude-opus-5" if use_advanced else "claude-haiku-4-5-20251001"

            try:
                with client.messages.stream(
                    model=model,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                ) as stream:
                    st.write_stream(stream.text_stream)
            except Exception as e:
                st.error(f"Claude request failed. Check your API key. ({e})")