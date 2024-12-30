import requests
from typing import List

from src.utils.config_loader import ConfigLoader
from src.llm.base_llm import BasePromptGenerator

config = ConfigLoader()


class OllamaClient(BasePromptGenerator):
    def __init__(self, base_url="http://localhost:11434/api"):
        self.base_url = base_url

    def generate_interpretation(self, cards_name: List[str], context: str = "") -> str:
        prompt = self.generate_input(cards_name, context)

        payload = {
            "model": config.get("llm", "model"), 
            "prompt": prompt,
            "stream": False,
        }

        response = requests.post(f"{self.base_url}/generate", json=payload)
        if response.status_code == 200:
            json_response = response.json()
            return json_response.get("response", "No interpretation")
        else:
            raise Exception(f"Error {response.status_code}: {response.text}")
