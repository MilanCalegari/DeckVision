import requests

from src.utils.config_loader import ConfigLoader

config = ConfigLoader()


class OllamaClient:
    def __init__(self, base_url="http://localhost:11434/api"):
        self.base_url = base_url

    def generate_interpretation(self, card_name: str, context: str = ""):
        prompt = f"""
        You are a Tarot expoert. Interpret the '{card_name}' in the context of a reding.
        Context: {context}
        Provide a detailed but concise interpretation
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
