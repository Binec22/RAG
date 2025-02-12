import logging
import os
from dotenv import load_dotenv
from typing import Callable, Dict

from langchain_voyageai import VoyageAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings


class CustomEmbeddings:
    def __init__(self, model_name: str ="voyage-3"):
        """Initialize the Embeddings object with a default model name and load the specific API key if necessary."""
        self._model_name = model_name
        self.model = self.load_model()

    @property
    def model_name(self) -> str:
        """Get the current model name."""
        return self._model_name

    @model_name.setter
    def model_name(self, value: str) -> None:
        """Set a new model name."""
        if not isinstance(value, str):
            raise ValueError("Le nom du modèle doit être une chaîne")
        self._model_name = value

    def load_api_key(self) -> None:
        """Load the specific API key for the model from environment variables or set a fixed key for 'voyage' models."""
        # Load environment variables from the .env file
        load_dotenv()

        try:
            if "voyage" in self.model_name.lower():
                api_key = os.getenv("VOYAGE_API_KEY")
                if not api_key:
                    raise APIKeyMissingError(self.model_name)
                os.environ["VOYAGE_API_KEY"] = api_key

            elif "openai" in self.model_name.lower():
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise APIKeyMissingError(self.model_name)
                os.environ["OPENAI_API_KEY"] = api_key
        except Exception as e:
            raise InvalidAPIKeyError(self.model_name, str(e))

    @classmethod
    def create_embedding_model(cls, model_name: str) -> Callable:
        """Factory method to create embedding models."""
        embeddings_map: Dict[str, Callable] = {
            "sentence-transformers/all-mpnet-base-v2": lambda: HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2"),
            "openai": OpenAIEmbeddings,
            "voyage-3": lambda: VoyageAIEmbeddings(model="voyage-3"),
            "voyage-law-2": lambda: VoyageAIEmbeddings(model="voyage-law-2"),
            "voyage-multilingual-2": lambda: VoyageAIEmbeddings(model="voyage-multilingual-2"),
            "nomic-embed-text": lambda: OllamaEmbeddings(model="nomic-embed-text")
        }

        if model_name not in embeddings_map:
            raise ValueError(f"Modèle {model_name} non supporté")

        return embeddings_map[model_name]()

    def load_model(self):
        """Load the embedding model based on the specified name."""
        try:
            self.load_api_key()
            return self.create_embedding_model(self.model_name)
        except Exception as e:
            logging.error(f"Erreur lors du chargement du modèle : {e}")
            raise


class APIKeyMissingError(Exception):
    """Exception raised when a specific model API key is missing in environment variables."""
    def __init__(self, model_name: str, message: str = "API key is missing in environment"):
        self.model_name = model_name
        self.message = f"{message} for model: {model_name}"
        super().__init__(self.message)


class InvalidAPIKeyError(Exception):
    """Exception raised when a specific model API key is invalid in environment variables."""

    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.api_key = api_key
        self.message = f"{model_name}'s API key is invalid : {api_key} "
        super().__init__(self.message)
