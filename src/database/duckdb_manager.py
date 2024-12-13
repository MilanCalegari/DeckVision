from dataclasses import dataclass
from typing import List, Tuple

import duckdb
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.features.feature_extractor import FeatureExtractor
from src.utils.config_loader import ConfigLoader

config = ConfigLoader()


@dataclass
class Card:
    name: str
    feature: np.ndarray



class DuckDataBase:
    def __init__(self) -> None:
        self.db_path = config.get("database", "path")
        self.conn = duckdb.connect(self.db_path)
        self.feature_extractor = FeatureExtractor()

    def get_all_features(self) -> List[Card]:
        query = "SELECT card_name, features FROM card_features;"
        results = self.conn.execute(query).fetchall()

        cards = []
        for card_name, feature_blob in results:
            feature_array = np.frombuffer(feature_blob, dtype=np.float32)
            cards.append(Card(name=card_name, feature=feature_array))
        return cards

    def find_most_similar_card(self, card_img_path: str) -> Tuple[str, float]:
        input_features = self.feature_extractor.extract_features(card_img_path)
        
        cards = self.get_all_features()

        min_distance = 0
        most_similar_card = None

        for card in cards:
            card_name = card.name
            card_feature = card.feature

            distance = cosine_similarity([card_feature], [input_features]) 
            print(f"Card: {card_name} Distance: {distance}\n")
            if distance > min_distance:
                min_distance = distance
                most_similar_card = card_name
    
        return most_similar_card, min_distance
