from langchain_community.document_loaders import JSONLoader
from langchain_experimental.text_splitter import SemanticChunker

import sys
import os
# Ajoute le chemin racine du projet au PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from RagClasses.Embedding import Embeddings

emd_model = Embeddings(model_name="voyage-3").model

json_loader = JSONLoader(
    file_path="C:\\Users\\Antonin\\PycharmProjects\\RAG\\data\\seatech_reformule\\documents\\base_de_donnees_reformulee.json",
    jq_schema=".[]",
    content_key="markdown",
)
text_splitter = SemanticChunker(emd_model)
docs = json_loader.load()
splitted_docs = text_splitter.split_documents(docs[0:2])

print(docs[0])
print("\n")
print(splitted_docs[0])
print("\n")
print(len(splitted_docs))
print("\n")
print(splitted_docs)
