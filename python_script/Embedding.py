import os
from dotenv import load_dotenv


class Embeddings:
    def __init__(self, model_name="voyage-3", load=True):
        """Initialise l'objet Embeddings avec un nom de modèle par défaut et charge la clé API spécifique si
        nécessaire."""
        self.model_name: str = model_name
        if load:
            self.embeddings = self.load_model()
        else:
            self.embeddings = None

    def load_api_key(self):
        """Charge la clé API spécifique au modèle depuis les variables d'environnement ou fixe la clé pour les
        modèles 'voyage'."""
        load_dotenv()  # Charge les variables d'environnement depuis le fichier .env
        api_key = None

        # Si le modèle contient "voyage", utiliser une clé fixe
        if "voyage" in self.model_name.lower():
            if not os.getenv("VOYAGE_API_KEY"):
                raise APIKeyMissingError(self.model_name)
            os.environ["VOYAGE_API_KEY"] = os.getenv("VOYAGE_API_KEY")

        elif "openai" in self.model_name.lower():
            if not os.getenv("OPENAI_API_KEY"):
                raise APIKeyMissingError(self.model_name)
            os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

    def load_model(self):
        """Charge le modèle d'embeddings en fonction du nom spécifié."""
        # Vérification de la clé API avant de charger le modèle
        self.load_api_key()

        # Logique pour charger le modèle
        if self.model_name == "sentence-transformers/all-mpnet-base-v2":
            from langchain_community.embeddings import HuggingFaceEmbeddings
            self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

        elif self.model_name == "openai":
            from langchain_openai import OpenAIEmbeddings
            self.embeddings = OpenAIEmbeddings()

        elif self.model_name == "voyage-3":
            from langchain_voyageai import VoyageAIEmbeddings
            self.embeddings = VoyageAIEmbeddings(model="voyage-3")

        elif self.model_name == "voyage-law-2":
            from langchain_voyageai import VoyageAIEmbeddings
            self.embeddings = VoyageAIEmbeddings(model="voyage-law-2")

        elif self.model_name == "voyage-multilingual-2":
            from langchain_voyageai import VoyageAIEmbeddings
            self.embeddings = VoyageAIEmbeddings(model="voyage-multilingual-2")

        else:
            raise ValueError("Invalid model name. Please choose a valid model.")

        return self.embeddings


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
