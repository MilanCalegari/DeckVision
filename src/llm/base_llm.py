from abc import ABC, abstractmethod
from typing import List


class BaseLLM(ABC):
    @abstractmethod
    def generate_interpretation(self, cards_name: List[str], context: str = "") -> str: ...

    @abstractmethod
    def generate_input(self, cards_name: List[str], context: str = "") -> str: ...


class BasePromptGenerator(BaseLLM):
    def generate_input(self, cards_name: List[str], context: str = "") -> str:
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
        """

        if len(cards_name) == 1:
            return f"""
            You are a Tarot expert. Interpret the '{cards_name[0]}' in the context of a reading.
            {general_prompt}
            """
        else:
            return f"""
            You are a Tarot expert. Interpret the following cards in the context of a reading: Past {cards_name[0]}, Present {cards_name[1]}, Future {cards_name[2]}.
            {general_prompt}
            """
