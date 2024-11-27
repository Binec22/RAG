import logging
import os
from dotenv import load_dotenv

from langchain_voyageai import VoyageAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings


class Embeddings:
    def __init__(self, model_name="voyage-3"):
        """Initialise l'objet Embeddings avec un nom de modèle par défaut et charge la clé API spécifique si
        nécessaire."""
        self._model_name = model_name
        self.model = self.load_model()

    @property
    def model_name(self):
        return self._model_name

    @model_name.setter
    def model_name(self, value):
        if not isinstance(value, str):
            raise ValueError("Le nom du modèle doit être une chaîne")
        self._model_name = value

    def load_api_key(self):
        """Charge la clé API spécifique au modèle depuis les variables d'environnement ou fixe la clé pour les
        modèles 'voyage'."""
        load_dotenv()  # Charge les variables d'environnement depuis le fichier .env

        # Si le modèle contient "voyage", utiliser une clé fixe
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
    def create_embedding_model(cls, model_name):
        """Méthode de factory pour créer des modèles d'embeddings"""
        embeddings_map = {
            "sentence-transformers/all-mpnet-base-v2": lambda: HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2"),
            "openai": OpenAIEmbeddings,
            "voyage-3": lambda: VoyageAIEmbeddings(model="voyage-3"),
            "voyage-law-2": lambda: VoyageAIEmbeddings(model="voyage-law-2"),
            "voyage-multilingual-2": lambda: VoyageAIEmbeddings(model="voyage-multilingual-2")
        }

        if model_name not in embeddings_map:
            raise ValueError(f"Modèle {model_name} non supporté")

        return embeddings_map[model_name]()

    def load_model(self):
        """Charge le modèle d'embeddings en fonction du nom spécifié."""
        try:
            self.load_api_key()
            return self.create_embedding_model(self.model_name)
        except Exception as e:
            logging.error(f"Erreur lors du chargement du modèle : {e}")
            raise


class APIKeyMissingError(Exception):
    """Exception levée lorsqu'une clé API spécifique au modèle est manquante dans les variables d'environnement."""

    def __init__(self, model_name, message="API key is missing in environment"):
        self.model_name = model_name
        self.message = f"{message} for model: {model_name}"
        super().__init__(self.message)


class InvalidAPIKeyError(Exception):
    """Exception levée lorsqu'une clé API spécifique au modèle est invalide dans les variables d'environnement."""

    def __init__(self, model_name, api_key):
        self.model_name = model_name
        self.api_key = api_key
        self.message = f"{model_name}'s API key is invalid : {api_key} "
        super().__init__(self.message)
