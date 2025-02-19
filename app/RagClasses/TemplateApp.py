import json
import os
from typing import Dict, Any

from flask import Flask, render_template, request, jsonify, send_from_directory

from app.RagClasses import RagChain, ConversationalRagChain, AppConfig

TEMPLATE_PATH = '/local.html'

class TemplateApp:
    def __init__(self, name: str, config: Dict[str, Any] = None):
        """
        Initialize the TemplateApp with a given name and configuration.

        Args:
            name (str): The name of the Flask app.
            config (Dict[str, Any]): Configuration dictionary for the app.
        """
        # Build absolute paths for the templates and static directories
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        template_folder = os.path.join(base_dir, 'templates')
        static_folder = os.path.join(base_dir, 'static')

        self.app = Flask(name, template_folder=template_folder, static_folder=static_folder)
        self.config = AppConfig(config=config, onLoad=True)
        self.config_dict = self.config.as_dict()
        self.data_path = self.config_dict.get("data_files_path")
        self._data_files = os.listdir(self.data_path)
        self.rag_chain = None
        self.conversational_rag_chain = None
        self.load_rag()
        self._setup_routes()

    def load_rag(self):
        """
        Load the RAG chain and conversational RAG chain with the current configuration
        """
        self.rag_chain = RagChain(config=self.config_dict)
        self.conversational_rag_chain = ConversationalRagChain.from_llm(
            rag_chain=self.rag_chain.rag_chain,
            llm=self.rag_chain.llm_model,
            callbacks=None
        )

    def _setup_routes(self):
        """
        Set up the routes for the Flask app
        """
        self.app.add_url_rule('/', view_func=self.index)
        self.app.add_url_rule('/documents/<document_name>', view_func=self.get_document, methods=['GET'])
        self.app.add_url_rule('/documents', view_func=self.list_documents, methods=['GET'])
        self.app.add_url_rule('/files/<filename>', view_func=self.serve_file, methods=['GET'])
        self.app.add_url_rule('/update-settings', view_func=self.update_settings, methods=['POST'])
        self.app.add_url_rule('/clear_chat_history', view_func=self.clear_chat_history, methods=['POST'])
        self.app.add_url_rule('/get', view_func=self.get_chat_response, methods=['POST'])

    def get_document(self, document_name: str):
        """
        Get details of a specific document.

        Args:
            document_name (str): The name of the document to retrieve.

        Returns:
            JSON response with document details or a 404 error if not found.
        """
        documents = [
            {"name": file, "url": f"/files/{file}", "extension": os.path.splitext(file)[1][1:]}
            for file in self._data_files
        ]
        document = next((doc for doc in documents if doc["name"] == document_name), None)

        if document is None:
            return jsonify({'error': 'Document not found'}), 404

        return jsonify(document)

    def run(self, **kwargs):
        """
        Run the Flask app with the specified configuration
        """
        self.app.run(**kwargs)

    def list_documents(self):
        """
        List all available documents.

        Returns:
            JSON response with a list of documents.
        """
        documents = [{"name": f, "url": f"/files/{f}", "extension": os.path.splitext(f)[1][1:]} for f in self._data_files]
        return jsonify(documents)

    def serve_file(self, filename: str):
        """
        Serve a file from the upload directory.

        Args:
            filename (str): The name of the file to serve.

        Returns:
            The file content.
        """
        return send_from_directory(self.app.config['UPLOAD_FOLDER'], filename)

    @staticmethod
    def index():
        """Render the main template."""
        return render_template(TEMPLATE_PATH)

    def update_settings(self):
        """
        Update the app settings with new configuration data.

        Returns:
            JSON response indicating success or failure.
        """
        data = request.get_json()
        self.config.update_settings(data)
        self.config_dict = self.config.as_dict()
        self.load_rag()
        return jsonify({'status': 'success', 'message': 'Settings updated successfully'}), 200

    def clear_chat_history(self):
        """
        Clear the chat history.

        Returns:
            JSON response indicating success.
        """
        self.conversational_rag_chain.clear_chat_history()
        return jsonify({'status': 'success', 'message': 'Chat history cleared'}), 200

    def get_chat_response(self):
        """
        Get a response from the conversational RAG chain based on a user query.

        Returns:
            JSON response with the chat response, context, and source.
        """
        data = request.get_json()
        query = data.get("msg", "")
        inputs = {
            "query": str(query),
            "chat_history": []
        }

        result = self.conversational_rag_chain._call(inputs)

        output_data = {
            'response': result['result'],
            'context': result['context'],
            'source': result['source']
        }
        return jsonify(output_data)
