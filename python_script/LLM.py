import os
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from typing import Optional


class LLM:
    _SUPPORTED_MODELS = {
        "gpt": lambda self: self._load_openai_model(),
        "mistral": lambda self: self._load_mistral_model()
    }

    def __init__(
            self,
            model_name: Optional[str] = None,
            cache_dir: Optional[str] = None,
            load: bool = True,
            temperature: float = 0
    ):
        if not 0 <= temperature <= 1:
            raise ValueError(f"Temperature must be between 0 and 1, got {temperature}")

        self._model_name = model_name
        self._cache_dir = cache_dir
        self._hf_token = None
        self._model = None
        self._temperature = temperature

        if load:
            self.load_model()

    @property
    def model_name(self):
        return self._model_name

    @property
    def model(self):
        return self.model

    def __load_api_key(self):
        """Charge les clés API depuis les variables d'environnement"""
        load_dotenv()

        if "openai" in self.model_name.lower():
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise APIKeyMissingError(self.model_name)
            os.environ["OPENAI_API_KEY"] = api_key

        elif "mistralai" in self.model_name.lower():
            hf_token = os.getenv("HF_API_TOKEN")
            if not hf_token:
                raise APIKeyMissingError(self.model_name)
            self._hf_token = hf_token
            os.environ["HF_API_TOKEN"] = hf_token

    def _load_openai_model(self):
        """Charge un modèle OpenAI"""
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model_name=self.model_name, temperature=self._temperature)

    def _load_mistral_model(self):
        """Charge un modèle Mistral"""
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            token=self._hf_token,
            cache_dir=self._cache_dir
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            token=self._hf_token,
            cache_dir=self._cache_dir
        )

        text_generation_pipeline = pipeline(
            model=model,
            tokenizer=tokenizer,
            task="text-generation",
            temperature=self._temperature,
            repetition_penalty=1.1,
            return_full_text=True,
            max_new_tokens=1000,
        )
        return HuggingFacePipeline(pipeline=text_generation_pipeline)

    def load_model(self):
        """Charge le modèle en fonction de son nom"""
        self.__load_api_key()

        for model_type, loader in self._SUPPORTED_MODELS.items():
            if model_type in self.model_name.lower():
                self.model = loader(self)
                return

        raise UnknownModelException(self.model_name)


class APIKeyMissingError(Exception):
    def __init__(self, model_name: str):
        super().__init__(f"API key is missing for model: {model_name}")


class UnknownModelException(Exception):
    def __init__(self, model_name: str):
        super().__init__(
            f"{model_name} is unknown. "
            "Verify the model name or ask an Admin to add this model to the system"
        )