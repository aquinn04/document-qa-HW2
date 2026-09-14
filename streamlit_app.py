import streamlit as st
from openai import OpenAI

# Define each page, pointing to its actual file path
hw1_page = st.Page("Hw1.py", title="HW 1", icon="📝")
hw2_page = st.Page("Hw2.py", title="HW 2", icon="🌐")

# Register them with navigation
pg = st.navigation([hw1_page, hw2_page])

st.set_page_config(page_title="HW Manager", page_icon="🧑‍💻")

pg.run()