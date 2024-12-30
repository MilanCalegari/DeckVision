from typing import List

import torch
from transformers import pipeline
from huggingface_hub import login

from src.llm.base_llm import BasePromptGenerator
from src.utils.config_loader import ConfigLoader

config = ConfigLoader()

class TransformersClient(BasePromptGenerator):
    def __init__(self):
        login(token=config.get("huggingface", "token"))

        self.pipeline = pipeline(
            "text-generation",
            model="meta-llama/Llama-3.2-1B-Instruct", 
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def generate_input(self, cards_name: List[str], context: str = "") -> str:
        content = f"""
        You are a tarot reader. You are given a set of tarot cards and you need to provide a reading for the user based on the cards.
        
        Instructions for Tarot Reading:

        General Instructions:
            - Answer must be short and concise.
            - Give a final summarized overview of the reading.
            - Only use the cards reversed if its provided as reversed.
            - If the cards are reversed, use the reversed meaning.

        1. If context is provided:
           - Focus primarily on interpreting within the given context
           - Keep the symbolism of the cards first, but make sure to use the context to interpret the cards.
        
        2. If no context is provided:
           - Provide a general daily life reading
           - Focus on practical matters and personal growth
           - Focus on the cards symbolism
           - Daily life situations, work, and personal development

        """

        messages = [
            { "role": "system", "content": content },
            { "role": "user", "content": f"The provided cards are: {cards_name}. The context of the question is: {context}" }
        ]

        return messages 


    def generate_interpretation(self, cards_name: List[str], context: str = "") -> str:
        messages = self.generate_input(cards_name, context)

        output = self.pipeline(messages, max_new_tokens=1024)

        return output[0]['generated_text'][-1]['content']
