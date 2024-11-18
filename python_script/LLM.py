from dotenv import load_dotenv
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline
)
from langchain_huggingface import HuggingFacePipeline

import os


class LLM:
    def __init__(self, model_name: str = None, cache_dir: str = None, load: bool = True, temperature: float = 0):
        self.model_name = model_name
        self.cache_dir = cache_dir if cache_dir else None
        self.hf_token = None
        self.llm_model = None
        self.temperature = temperature
        if load:
            self.load_model()
        if 0 < temperature > 1:
            raise TemperatureInvalidValue(temperature)

    def load_api_key(self):
        """Charge la clé API spécifique au modèle depuis les variables d'environnement ou fixe la clé pour les
        modèles 'voyage'."""
        load_dotenv()  # Charge les variables d'environnement depuis le fichier .env
        api_key = None

        if "openai" in self.model_name.lower():
            if not os.getenv(f"OPENAI_API_KEY"):
                raise APIKeyMissingError(self.model_name)
            os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

        elif "mistralai" in self.model_name.lower():
            if not os.getenv(f"OPENAI_API_KEY"):
                raise APIKeyMissingError(self.model_name)
            os.environ["HF_API_TOKEN"] = os.getenv("HF_API_TOKEN")
            self.hf_token = os.getenv("HF_API_TOKEN")

    def load_model(self):
        self.load_api_key()
        if "gpt" in self.model_name:
            from langchain_openai import ChatOpenAI
            model = ChatOpenAI(model_name=self.model_name, temperature=self.temperature)
            self.llm_model = model

        elif "mistral" in self.model_name.lower():
            if self.llm_model is None:
                tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=self.model_name,
                                                          trust_remote_code=True,
                                                          token=self.hf_token,
                                                          cache_dir=self.cache_dir)
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.padding_side = "right"

                model = AutoModelForCausalLM.from_pretrained(
                    pretrained_model_name_or_path=self.model_name,
                    token=self.hf_token,
                    cache_dir=self.cache_dir
                )

                text_generation_pipeline = pipeline(
                    model=model,
                    tokenizer=tokenizer,
                    task="text-generation",
                    temperature=self.temperature,
                    repetition_penalty=1.1,
                    return_full_text=True,
                    max_new_tokens=1000,
                )
                self.llm_model = HuggingFacePipeline(pipeline=text_generation_pipeline)

        else:
            raise UnknownModelException(self.model_name)


class APIKeyMissingError(Exception):
    """Exception levée lorsqu'une clé API spécifique au modèle est manquante dans les variables d'environnement."""

    def __init__(self, model_name: str, message: str = "API key is missing in environment"):
        self.model_name = model_name
        self.message = f"{message} for model: {model_name}"
        super().__init__(self.message)


class TemperatureInvalidValue(ValueError):
    """ValueError levée lorsque la température n'est pas un float compris entre 0 et 1"""

    def __init__(self, temperature: float, message="Temperature must be float between 0 and 1"):
        self.temperature = temperature
        self.message = f"{message}, current value is: {temperature}"
        super().__init__(self.message)


class UnknownModelException(Exception):
    """Exception levée lorsqu'un modèle ne correspond à aucune des modèles pré-enregistrés dans le système"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.message = (f"{model_name} is unknown\n"
                        f"Verify the model name or ask an Admnin to add this model to the system")
        super().__init__(self.message)
