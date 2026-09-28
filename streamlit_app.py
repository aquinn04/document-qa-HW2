import streamlit as st
from openai import OpenAI

# Define each page, pointing to its actual file path
hw2_page = st.Page("Hw2.py", title="HW 2", icon="🌐")
hw3_page = st.Page("Hw3.py", title="HW 3", icon="💬")
hw4_page = st.Page("Hw4.py", title="HW 4", icon="🤖")
hw5_page = st.Page("Hw5.py", title="HW 5", icon="🧑‍💻", default=True)

# Register them with navigation
pg = st.navigation([hw2_page, hw3_page, hw4_page, hw5_page])

st.set_page_config(page_title="HW Manager", page_icon="🧑‍💻")

pg.run()
