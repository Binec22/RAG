from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_from_directory
from RagChain import RagChain
from ConversationalRagChain import ConversationalRagChain
import os

class TemplateApp:
    def __init__(self, name, config=None):
        self.app = Flask(name)
        self.config = None
        self.data_path = None
        self._data_files = None
        self.rag_chain = None
        self.conversational_rag_chain = None
        self.load_config(config=config)
        self.load_rag()
        self._setup_routes()

    def load_config(self, config):
        if config is None or not isinstance(config, dict) or config == {}:
            raise ValueError("A valid configuration dictionary is required.")

        # Paramètres nécessaires
        needed_params = {
            "data_files_path": str,
            "embedded_database_path": str,
            "embedding_model": str,
            "llm_model": str,
        }

        # Vérification des clés nécessaires
        missing_keys = [key for key in needed_params if key not in config]
        if missing_keys:
            for key in missing_keys:
                print(f'Key "{key}" is required to launch the app.')
            raise ValueError(f"Missing configuration keys: {', '.join(missing_keys)}")

        # Vérification des types des clés nécessaires
        for key, expected_type in needed_params.items():
            if not isinstance(config[key], expected_type):
                raise TypeError(
                    f'Key "{key}" must be of type {expected_type.__name__}, '
                    f'but got {type(config[key]).__name__}.'
                )

        # Paramètres par défaut
        default_params = {
            "search_type": "similarity",
            "similarity_doc_nb": 5,
            "score_threshold": 0.8,
            "max_chunk_return": 5,
            "considered_chunk": 25,
            "mmr_doc_nb": 5,
            "lambda_mult": 0.25,
            "isHistoryOn": True,
        }

        # Fusion des configurations
        self.config = {**default_params, **config}

        self.data_path = self.config["data_files_path"]
        self._data_files = os.listdir(self.config["data_files_path"])

        # Confirmation de chargement
        print("Configuration loaded successfully.")

    def load_rag(self):
        #TODO
        self.rag_chain = RagChain(parameters=self.config)
        self.conversational_rag_chain = ConversationalRagChain.from_llm(
            rag_chain=self.rag_chain.rag_chain,
            llm=self.rag_chain.llm_model,
            callbacks=None
        )

    def _setup_routes(self):
        self.app.add_url_rule(
            '/',
            view_func=self.index,
        )
        self.app.add_url_rule(
            '/documents/<document_name>',
            view_func=self.get_document,
            methods=['GET']
        )
        self.app.add_url_rule(
            '/documents',
            view_func=self.list_documents,
            methods=['GET']
        )
        self.app.add_url_rule(
            '/files/<filename>',
            view_func=self.serve_file,
            methods=['GET']
        )
        self.app.add_url_rule(
            '/update-settings',
            view_func=self.update_settings,
            methods=['POST']
        )
        self.app.add_url_rule(
            '/clear_chat_history',
            view_func=self.clear_chat_history,
            methods=['POST']
        )

    def get_document(self, document_name):
        documents = [
            {"name": file, "url": f"/files/{file}", "extension": os.path.splitext(file)[1][1:]}
            for file in self._data_files
        ]
        document = next((doc for doc in documents if doc["name"] == document_name), None)

        if document is None:
            return jsonify({'error': 'Document not found'}), 404

        return jsonify(document)
    
    def run(self, **kwargs):
        self.app.run(**kwargs)

    def list_documents(self):
        documents = [{"name": f, "url": f"/files/{f}", "extension": os.path.splitext(f)[1][1:]} for f in self.data_files]
        return jsonify(documents)

    def serve_file(self, filename):
        return send_from_directory(self.app.config['UPLOAD_FOLDER'], filename)

    @staticmethod
    def index():
        return render_template('local.html')

    def get_Chat_response(self, query):
        inputs = {
            "query": str(query),
            "chat_history": []
        }
        res = self.conversational_rag_chain._call(inputs)

        output = jsonify({
            'response': res['result'],
            'context': res['context'],
            'source': res['source']
        })
        return output

    def update_settings(self):
        data = request.get_json()
        self.load_config(data)
        self.load_rag()
        return jsonify({'status': 'success', 'message': 'Settings updated successfully'}), 200

    def clear_chat_history(self):
        self.conversational_rag_chain.clear_chat_history()
        return jsonify({'status': 'success', 'message': 'Chat history cleared'}), 200

