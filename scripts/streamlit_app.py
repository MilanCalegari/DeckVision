import os

import streamlit as st

from src.database.duckdb_manager import DuckDataBase
from src.llm.ollama_client import OllamaClient
from src.segmentation.card_segmentation import CardSegmentation
from src.utils.config_loader import ConfigLoader

config = ConfigLoader()
llm_client = OllamaClient()
card_segmentation = CardSegmentation()
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

        cards_name = []
        # Identify the card
        with st.spinner("🔮 Summoning ancient spirits..."):
            cards = card_segmentation.run(file_path)
            with st.spinner("🌙 Aligning the cosmic energies..."):
                for card in cards:
                    with st.spinner("✨ Consulting my crystal ball and third eye..."):
                        most_similar_card, similarity = db.find_most_similar_card(card)
                        cards_name.append(most_similar_card)   
        st.image(file_path, caption=f"Uploaded Image: {[name for name in cards_name]}", use_container_width=True)

        os.remove(file_path)

        # Generate interpretation
        with st.spinner("Generating Tarot reading..."):
            try:
                interpretation = llm_client.generate_interpretation(cards_name, context)
                st.success(f"The card is: **{cards_name}**")
                st.subheader("Tarot Reading")
                st.write(interpretation)
            except Exception as e:
                st.error("Error generating interpretation")
