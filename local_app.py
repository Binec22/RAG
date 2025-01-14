import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'RagClasses'))
from RagClasses.TemplateApp import TemplateApp

default_params = {
    "temperature": 0,
    "search_type": "similarity",
    "similarity_doc_nb": 5,
    "score_threshold": 0.8,
    "max_chunk_return": 5,
    "considered_chunk": 25,
    "mmr_doc_nb": 5,
    "lambda_mult": 0.25,
    "isHistoryOn": True,
    "data_files_path": "data/test/documents/",
    "embedded_database_path": "data/test/embedded_database/",
    "embedding_model": "voyage-3",
    "llm_model": "gpt-3.5-turbo",
}


if __name__ == '__main__':
    local_app = TemplateApp(__name__,default_params)
    local_app.run(port=5000 , debug=False)