import os
from dotenv import load_dotenv
from langchain_community.callbacks.manager import openai_callback_var
from nltk.chat.iesha import reflections
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from typing import Optional


class LLM:
    _SUPPORTED_MODELS = {
        "gpt": lambda self: self._load_openai_model(),
        "hugging-face-mistral": lambda self: self._load_mistral_model(),
        "ollama-mistral": lambda self: self._load_ollama_mistral_model(),
        "ollama-deepseek": lambda self: self._load_ollama_deepseek_model(),
        "groq-mistral": lambda self: self._load_groq_mistral_model(),
        "groq-deepseek": lambda self: self._load_groq_deepseek_model(),
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

        self.model_name = model_name
        self._cache_dir = cache_dir
        self._hf_token = None
        self.model = None
        self._temperature = temperature
        if load:
            self.load_model()

    def __load_api_key(self):
        """Charge les clés API depuis les variables d'environnement"""
        load_dotenv()

        if "openai" in self.model_name.lower():
            openai_api_key = os.getenv("OPENAI_API_KEY")
            print(openai_api_key)
            if not openai_api_key:
                raise APIKeyMissingError(self.model_name)
            os.environ["OPENAI_API_KEY"] = openai_api_key

        elif "hugging-face" in self.model_name.lower():
            hf_api_key = os.getenv("HF_API_TOKEN")
            if not hf_api_key:
                raise APIKeyMissingError(self.model_name)
            self._hf_token = hf_api_key
            os.environ["HF_API_TOKEN"] = hf_api_key

        elif "groq" in self.model_name.lower():
            groq_api_key = os.getenv("GROQ_API_KEY")
            if not groq_api_key:
                raise APIKeyMissingError(self.model_name)
            self._groq_api_key = groq_api_key
            os.environ["GROQ_API_KEY"] = groq_api_key

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

    def _load_ollama_mistral_model(self):
        from langchain_ollama import ChatOllama
        print("le modèle Mistral 7B s'apprête à être chargé avec Ollama")
        return ChatOllama(model="mistral",
                          temperature=self._temperature,)

    def _load_ollama_deepseek_model(self):
        from langchain_ollama import ChatOllama
        print("le modèle DeepSeek s'apprête à être chargé avec Ollama")
        return ChatOllama(model="deepseek-r1",
                          temperature=self._temperature,)

    def _load_groq_mistral_model(self):
        from langchain_groq import ChatGroq
        print("le modèle Mistral 8x7b s'apprête à être chargé avec l'API Groq")
        return ChatGroq(model_name="mixtral-8x7b-32768",
                        temperature=self._temperature,)

    def _load_groq_deepseek_model(self):
        from langchain_groq import ChatGroq
        print("le modèle DeepSeek s'apprête à être chargé avec l'API Groq")
        return ChatGroq(model_name="deepseek-r1-distill-llama-70b-specdec",
                        temperature=self._temperature,)


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