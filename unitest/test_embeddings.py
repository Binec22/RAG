import unittest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../python_script')))

from Embedding import Embeddings, APIKeyMissingError, InvalidAPIKeyError


class TestEmbeddings(unittest.TestCase):

    def test_load_model_valid_model(self):
        """Test que le modèle est chargé correctement pour un modèle valide avec load=True (comportement par défaut)."""
        embeddings_instance = Embeddings(model_name="voyage-3")
        model = embeddings_instance.load_model()  # Charge explicitement le modèle
        self.assertIsNotNone(model, "Le modèle ne doit pas être None")

    def test_load_model_invalid_model(self):
        with self.assertRaises(ValueError):
            embeddings_instance = Embeddings(model_name="invalide")
            embeddings_instance.load_model()  # Tentative de chargement avec un modèle invalide

    def test_load_model_different_models(self):
        """Test la charge des différents modèles pris en charge avec load=True (comportement par défaut)."""
        models = [
            "openai",
            "voyage-3",
        ]
        for model_name in models:
            with self.subTest(model=model_name):
                embeddings_instance = Embeddings(model_name=model_name)
                model = embeddings_instance.load_model()
                self.assertIsNotNone(model)

    def test_load_model_with_load_false(self):
        """Test que le modèle n'est pas chargé avec load=False."""
        embeddings_instance = Embeddings(model_name="voyage-3", load=False)
        self.assertIsNone(embeddings_instance.model, "Le modèle ne doit pas être chargé immédiatement.")

        # Vérifier que load_model() charge correctement le modèle
        model = embeddings_instance.load_model()
        self.assertIsNotNone(model, "Le modèle doit être chargé après appel de load_model()")

    def test_load_model_invalid_model_with_load_false(self):
        """Test le chargement d'un modèle invalide avec load=False."""
        embeddings_instance = Embeddings(model_name="invalide", load=False)
        self.assertIsNone(embeddings_instance.model, "Le modèle ne doit pas être chargé immédiatement.")

        with self.assertRaises(ValueError):
            embeddings_instance.load_model()  # Tentative de chargement avec un modèle invalide


if __name__ == "__main__":
    unittest.main()
