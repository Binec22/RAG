from flask import Flask, render_template, request, jsonify, send_from_directory
from RagChain import RagChain
from ConversationalRagChain import ConversationalRagChain
from AppConfig import AppConfig
import os

class TemplateApp:
    def __init__(self, name, config=None):
        self.app = Flask(name)
        self.config = AppConfig(config=config).as_dict()
        self.data_path = self.config.get("data_files_path")
        self._data_files = os.listdir(self.data_path)
        self.rag_chain = None
        self.conversational_rag_chain = None
        self.load_rag()
        self._setup_routes()

    def load_rag(self):
        #TODO
        self.rag_chain = RagChain(config=self.config)
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
        self.app.add_url_rule(
            '/get',
            view_func=self.get_chat_response,
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
        documents = [{"name": f, "url": f"/files/{f}", "extension": os.path.splitext(f)[1][1:]} for f in self._data_files]
        return jsonify(documents)

    def serve_file(self, filename):
        return send_from_directory(self.app.config['UPLOAD_FOLDER'], filename)

    @staticmethod
    def index():
        return render_template('local.html')

    def update_settings(self):
        data = request.get_json()
        self.config = AppConfig(data).as_dict()
        self.load_rag()
        return jsonify({'status': 'success', 'message': 'Settings updated successfully'}), 200

    def clear_chat_history(self):
        self.conversational_rag_chain.clear_chat_history()
        return jsonify({'status': 'success', 'message': 'Chat history cleared'}), 200

    def get_chat_response(self):
        data = request.get_json()
        query = data.get("msg", "")
        inputs = {
            "query": str(query),
            "chat_history": []
        }

        result = self.conversational_rag_chain._call(inputs)

        output = jsonify({
            'response': result['result'],
            'context': result['context'],
            'source': result['source']
        })
        return output

