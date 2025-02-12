import json
from pathlib import Path
from pprint import pprint
from langchain_community.document_loaders import JSONLoader

input_json = "firecrawl_database5.json"

# Définir une fonction pour extraire des métadonnées spécifiques
def metadata_func(record: dict, metadata: dict) -> dict:
    # Extraire les métadonnées globales
    metadata["url"] = record.get("url")
    metadata["key_words"] = record.get("key_words")
    return metadata

# Initialiser le loader
loader = JSONLoader(
    file_path=input_json,
    jq_schema=".[] | .data",  # Garder tout l'objet JSON pour chaque entrée
    content_key="text_content",  # Extraire uniquement "data.data" comme contenu principal
    metadata_func=metadata_func,  # Ajouter les métadonnées nécessaires
    text_content=False
)
# Charger les documents
docs = loader.load()
print(docs[0])
print("\n\n")
print("\n\n")
#
# print("Contenu :", docs[0].page_content.encode('utf-8').decode('unicode_escape'))
print("\n\n")
print("\n\n")

print("Métadonnées :", docs[0].metadata)

