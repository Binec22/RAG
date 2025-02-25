import os

from app.RagClasses.TemplateApp import TemplateApp

default_params = {
    "database_name" : 'seatech_firecrawl',
    "temperature": 0,
    "search_type": "similarity",
    "similarity_doc_nb": 5,
    "score_threshold": 0.8,
    "max_chunk_return": 5,
    "considered_chunk": 25,
    "mmr_doc_nb": 5,
    "lambda_mult": 0.25,
    "isHistoryOn": True,
    "embedding_model": "voyage-3",
    "llm_model": "groq-mistral",
}

# Construire les chemins dynamiquement
base_path = os.path.join(os.path.dirname(__file__), 'data', default_params["database_name"])
default_params["data_files_path"] = os.path.join(base_path, 'documents')
default_params["embedded_database_path"] = os.path.join(base_path, 'embedded_database')



if __name__ == '__main__':
    local_app = TemplateApp(__name__, default_params)
    local_app.run(port=5000 , debug=True)