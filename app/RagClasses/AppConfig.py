import json
import os
from typing import Dict, Any, Optional

DEFAULT_CONFIG = 'config.json'

class AppConfig:
    def __init__(self, config: Dict[str, Any], onLoad : bool = False):
        """
        Initialize the AppConfig with a given configuration dictionary.

        Args:
            config (Dict[str, Any]): The configuration dictionary.
            onLoad (bool): Flag indicating if the configuration is being loaded initially.
        """
        # Default parameters with their default values
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
        # Required parameters with their expected types
        self.needed_params = {
            "data_files_path": str,
            "embedded_database_path": str,
            "embedding_model": str,
            "llm_model": str,
        }
        # Validate and merge the provided configuration with defaults
        self.config = self._validate_and_merge_config(config, onLoad)

    @classmethod
    def from_json(cls, file_path: Optional[str] = None, config_name: str = "default"):
        """
        Create an AppConfig object from a named configuration in a JSON file.

        Args:
            file_path (str): Path to the JSON configuration file.
            config_name (str): Name of the specific configuration to load.

        Returns:
            AppConfig: An initialized AppConfig object.
        """
        # Determine the file path
        if file_path is None:
            # default config path
            file_path = DEFAULT_CONFIG
            if not os.path.isfile(DEFAULT_CONFIG):
                raise FileNotFoundError(f"Default configuration file '{DEFAULT_CONFIG}' not found.")
        else:
            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"Configuration file '{file_path}' not found.")

        # Load the JSON file
        with open(file_path, "r") as file:
            all_configs = json.load(file)

        # Check if the specified configuration exists
        if config_name not in all_configs:
            raise ValueError(f"Configuration '{config_name}' not found in the file.")

        # Initialize the AppConfig with the selected configuration
        selected_config = all_configs[config_name]
        return cls(selected_config)

    def update_settings(self, config: Dict[str, Any]) -> None:
        """
        Update the configuration with new settings.

        Args:
            config (Dict[str, Any]): The new configuration settings to update.
        """
        if not isinstance(config, dict):
            raise ValueError("Configuration update must be a dictionary.")

        # Validate and update the configuration
        for key, value in config.items():
            if key in self.needed_params and not isinstance(value, self.needed_params[key]):
                raise TypeError(
                    f'Key "{key}" must be of type {self.needed_params[key].__name__}, '
                    f'but got {type(value).__name__}.'
                )

        self.config.update(config)

    def _validate_and_merge_config(self, config: Dict[str, Any], onLoad: bool) -> Dict[str, Any]:
        """
        Validate the configuration and merge it with default parameters.

        Args:
            config (Dict[str, Any]): The configuration dictionary to validate and merge.
            onLoad (bool): Flag indicating if the configuration is being loaded initially.

        Returns:
            Dict[str, Any]: The merged configuration dictionary.
        """
        if not isinstance(config, dict):
            raise ValueError("A valid configuration dictionary is required.")

        # Check for missing required keys
        if onLoad:
            missing_keys = [key for key in self.needed_params if key not in config]
            if missing_keys:
                raise ValueError(f"Missing configuration keys: {', '.join(missing_keys)}")

        # Validate types of required keys
        for key, expected_type in self.needed_params.items():
            if not isinstance(config[key], expected_type):
                raise TypeError(
                    f'Key "{key}" must be of type {expected_type.__name__}, '
                    f'but got {type(config[key]).__name__}.'
                )

        # Merge with default parameters
        return {**self.default_params, **config}

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """
        Get a configuration value by key.

        Args:
            key (str): The configuration key.
            default (Optional[Any]): The default value to return if the key is not found.

        Returns:
            Any: The value associated with the key, or the default value if the key is not found.
        """
        return self.config.get(key, default)

    def as_dict(self) -> Dict[str, Any]:
        """
        Get the configuration as a dictionary.

        Returns:
            Dict[str, Any]: The configuration dictionary.
        """
        return self.config
