import requests
from typing import List

from src.utils.config_loader import ConfigLoader

config = ConfigLoader()


class OllamaClient:
    def __init__(self, base_url="http://localhost:11434/api"):
        self.base_url = base_url

    def generate_interpretation(self, cards_name: List[str], context: str = ""):

        general_prompt = f"""
        Context: {context}

        Instructions for Tarot Reading:
        1. If context is provided:
           - Focus primarily on interpreting within the given context
           - Keep the symbolism of the cards first, but make sure to use the context to interpret the cards.
        
        2. If no context is provided:
           - Provide a general daily life reading
           - Focus on practical matters and personal growth
           - Focus on the cards symbolism
           - Daily life situations, work, and personal development
        
        3. Format guidelines:
           - Keep interpretation concise but meaningful
           - Use clear and direct language
           - Avoid lengthy explanations
        
        4. Important notes:
           - Do not default to love/relationship readings if no context is provided
           - Focus on daily life situations, work, and personal development
           - Maintain a practical and grounded perspective
           - Could teach a lesson to the user
        """


        if len(cards_name) == 1:
            prompt = f"""
            You are a Tarot expoert. Interpret the '{cards_name[0]}' in the context of a reading.
            {general_prompt}
            """
        else:
            prompt = f"""
            You are a Tarot expoert. Interpret the following cards in the context of a reading: Past {cards_name[0]}, Present {cards_name[1]}, Future {cards_name[2]}.
            {general_prompt}
            """

        payload = { 
            "model": config.get("llm", "model"),
            "prompt": prompt,
            "stream": False,
        }

        response = requests.post(f"{self.base_url}/generate", json=payload)
        if response.status_code == 200:
            json_response =  response.json()
            return json_response.get("response", "No interpretation")
        else:
            raise Exception(f"Error {response.status_code}: {response.text}")
