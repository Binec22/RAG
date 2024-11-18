import unittest
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../python_script')))
from LLM import LLM, APIKeyMissingError, UnknownModelException


class TestLLM(unittest.TestCase):

    @patch('LLM.load_dotenv')
    @patch('LLM.os.getenv')
    def test_load_api_key_missing_openai_key(self, mock_getenv, mock_load_dotenv):
        mock_getenv.return_value = None
        llm = LLM(model_name='openai-gpt', load=False)

        with self.assertRaises(APIKeyMissingError):
            llm.load_api_key()

    # @patch('LLM.load_dotenv')
    # @patch('LLM.os.getenv')
    # def test_load_api_key_present(self, mock_getenv, mock_load_dotenv):
    #     mock_getenv.side_effect = lambda key: 'fake_api_key' if key == 'OPENAI_API_KEY' else None
    #     llm = LLM(model_name='openai-gpt', load=False)
    #
    #     try:
    #         llm.load_api_key()
    #     except APIKeyMissingError:
    #         self.fail("APIKeyMissingError was raised unexpectedly!")

    @patch('LLM.AutoTokenizer.from_pretrained')
    @patch('LLM.AutoModelForCausalLM.from_pretrained')
    @patch('LLM.pipeline')
    def test_load_model_mistral(self, mock_pipeline, mock_model, mock_tokenizer):
        mock_pipeline.return_value = MagicMock()
        mock_tokenizer.return_value = MagicMock()
        mock_model.return_value = MagicMock()

        llm = LLM(model_name='mistral-test', cache_dir='/tmp', load=False)
        try:
            llm.load_model()
        except Exception as e:
            self.fail(f"Loading mistral model failed: {e}")

    def test_unknown_model_exception(self):
        llm = LLM(model_name='unknown-model', load=False)
        with self.assertRaises(UnknownModelException):
            llm.load_model()


if __name__ == '__main__':
    unittest.main()
