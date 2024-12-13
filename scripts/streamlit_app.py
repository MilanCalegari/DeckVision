import os

import streamlit as st

from src.database.duckdb_manager import DuckDataBase
from src.llm.ollama_client import OllamaClient
from src.utils.config_loader import ConfigLoader

config = ConfigLoader()
llm_client = OllamaClient()
db = DuckDataBase()

st.title("DeckVision: Tarot Reading")
st.write("Uploat a Tarot card image to identify it and recieve an interpretation")

uploaded_file = st.file_uploader("Choose an image...", type=config.get("streamlit", "allowed_files_types"))
context = st.text_area("Provide additional context for the reading (optional):", "")

if st.button("Get Tarot Reading"):
    if uploaded_file is not None:
        temp_path = config.get("paths", "temp_upload")
        os.makedirs(temp_path, exist_ok=True)
        file_path = os.path.join(temp_path, uploaded_file.name)
        # Write the card image
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Identify the card
        with st.spinner("Identifying the card..."):
            most_similar_card, similarity = db.find_most_similar_card(file_path)

        st.image(file_path, caption=f"Uploaded Image: {most_similar_card}", use_container_width=True)

        os.remove(file_path)

        # Generate interpretation
        with st.spinner("Generating Tarot reading..."):
            try:
                interpretation = llm_client.generate_interpretation(most_similar_card, context)
                st.success(f"The card is: **{most_similar_card}**")
                st.subheader("Tarot Reading")
                st.write(interpretation)
            except Exception as e:
                st.error("Error generating interpretation")
