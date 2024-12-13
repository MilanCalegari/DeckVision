import json

import duckdb
import numpy as np
from tqdm import tqdm

from src.features.feature_extractor import FeatureExtractor


class DuckDataBase:
    def __init__(self) -> None:
        self.feature_extractor = FeatureExtractor()
        self.db_path = "./db/features_db.duckdb"

        self.conn = duckdb.connect(self.db_path)
        self.conn.execute('''
        CREATE TABLE IF NOT EXISTS card_features (
            card_id INTEGER PRIMARY KEY, 
            card_name VARCHAR UNIQUE, 
            features BLOB
        );
        ''')
        self.write_cards("./data/")

    def write_cards(self, cards_path: str):
        img_folder = "./data/cards/"
        with open(f"{cards_path}/tarot-images.json", "r") as f:
            tarot_description = json.load(f)

        for idx, card_info in enumerate(tqdm(tarot_description['cards'])):
            card_name = card_info['name']
            card_img = card_info['img']
            feature = self.feature_extractor.extract_features(f"{img_folder}{card_img}").tobytes()
            # Write in db
            try:
                self.conn.execute('INSERT INTO card_features (card_id, card_name, features) VALUES (?, ?, ?)', (idx, card_name, feature))
            except Exception as e:
                print(f'Error inserting card {card_name}: {e}')



if __name__ == "__main__":
    db = DuckDataBase()
