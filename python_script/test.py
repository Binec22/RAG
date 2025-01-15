import json
from pathlib import Path
from pprint import pprint
from langchain_community.document_loaders import JSONLoader

input_json = "python_script/base_de_donnees_reformulee.json"

# Définir une fonction pour extraire des métadonnées spécifiques
def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["url"] = record.get("metadata", {}).get("url")
    metadata["title"] = record.get("metadata", {}).get("title")
    metadata["author"] = record.get("metadata", {}).get("author")
    metadata["language"] = record.get("metadata", {}).get("language")
    return metadata

# Initialiser le loader
loader = JSONLoader(
    file_path=input_json,  # Chemin vers votre fichier JSON
    jq_schema=".[]",                # Utilisation du chemin global (entièrement configurable)
    content_key="markdown",       # Clé utilisée pour extraire le contenu principal
    metadata_func=metadata_func   # Fonction pour extraire les métadonnées
)

# Charger les documents
docs = loader.load()

print("Contenu :", docs[0].page_content)
print("Métadonnées :", docs[0].metadata)

