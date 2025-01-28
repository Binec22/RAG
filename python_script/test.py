import json
from pathlib import Path
from pprint import pprint
from langchain_community.document_loaders import JSONLoader

input_json = "firecrawl_database4.json"

# Définir une fonction pour extraire des métadonnées spécifiques
def metadata_func(record: dict, metadata: dict) -> dict:
    # Ajouter l'URL des métadonnées au niveau racine
    metadata["url"] = record.get("url")
    return metadata

# Initialiser le loader
loader = JSONLoader(
    file_path=input_json,  # Chemin vers votre fichier JSON
    jq_schema=".[]",                # Utilisation du chemin global (entièrement configurable)
    content_key="data",             # Clé utilisée pour extraire le contenu principal
    metadata_func=metadata_func,
    text_content=False
)

# Charger les documents
docs = loader.load()
print(docs[0])

# print("Contenu :", docs[0].page_content.encode('utf-8').decode('unicode_escape'))
print("\n\n")
print("\n\n")

print("Métadonnées :", docs[0].metadata)

