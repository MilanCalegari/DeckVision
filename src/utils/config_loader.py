import yaml


class ConfigLoader:
    def __init__(self, config_path="./config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self):
        with open(self.config_path, "r") as file:
            return yaml.safe_load(file)

    def get(self, section, key=None):
        """Retrieve a specific section or key from the config"""
        if key:
            return self.config.get(section, {}).get(key)
        return self.config.get(section)

