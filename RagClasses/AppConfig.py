import json
import os


class AppConfig:
    def __init__(self, config : dict):
        self.default_params = {
            "temperature": 0,
            "search_type": "similarity",
            "similarity_doc_nb": 5,
            "score_threshold": 0.8,
            "max_chunk_return": 5,
            "considered_chunk": 25,
            "mmr_doc_nb": 5,
            "lambda_mult": 0.25,
            "isHistoryOn": True,
        }
        self.needed_params = {
            "data_files_path": str,
            "embedded_database_path": str,
            "embedding_model": str,
            "llm_model": str,
        }
        self.config = self._validate_and_merge_config(config)

    @classmethod
    def from_json(cls, file_path: str = None, config_name: str = "default"):
        """
        Create an AppConfig object from a named configuration in a JSON file.

        Args:
            file_path (str): Path to the JSON configuration file.
            config_name (str): Name of the specific configuration to load.

        Returns:
            AppConfig: An initialized AppConfig object.
        """
        if file_path is None:
            # default config path
            file_path = "config.json"
            if not os.path.isfile("config.json"):
                raise FileNotFoundError(f"Default configuration file 'config.json' not found.")
        else:
            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"Configuration file '{file_path}' not found.")

        with open(file_path, "r") as file:
            all_configs = json.load(file)

        if config_name not in all_configs:
            raise ValueError(f"Configuration '{config_name}' not found in the file.")

        selected_config = all_configs[config_name]
        return cls(selected_config)

    def _validate_and_merge_config(self, config):
        if not isinstance(config, dict):
            raise ValueError("A valid configuration dictionary is required.")

        # Vérification des clés nécessaires
        missing_keys = [key for key in self.needed_params if key not in config]
        if missing_keys:
            raise ValueError(f"Missing configuration keys: {', '.join(missing_keys)}")

        # Vérification des types
        for key, expected_type in self.needed_params.items():
            if not isinstance(config[key], expected_type):
                raise TypeError(
                    f'Key "{key}" must be of type {expected_type.__name__}, '
                    f'but got {type(config[key]).__name__}.'
                )

        # Fusion avec les paramètres par défaut
        return {**self.default_params, **config}

    def get(self, key, default=None):
        return self.config.get(key, default)

    def as_dict(self):
        return self.config
