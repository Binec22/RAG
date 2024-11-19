from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_from_directory
import os

class TemplateApp:
    def __init__(self, name, config=None):
        self.app = Flask(name)
        self.files = os.listdir(self.app.config['UPLOAD_FOLDER'])
        self._setup_routes()

    def load_rag(settings=None):
        #TODO
        global rag_conv
        rag_chain = RagChain(parameters=None)
        rag_conv = ConversationalRagChain.from_llm(
            rag_chain=rag_chain.rag_chain,
            llm=rag_chain.llm_model,
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
            view_func=self.clear_chat_history(),
            methods=['POST']
        )

    def get_document(self, document_name):
        #TODO define UPLOAD_FOLDER
        documents = [
            {"name": f, "url": f"/files/{f}", "extension": os.path.splitext(f)[1][1:]}
            for f in self.files
        ]

        document = next((doc for doc in documents if doc["name"] == document_name), None)

        if document is None:
            return jsonify({'error': 'Document not found'}), 404

        return jsonify(document)
    
    def run(self, **kwargs):
        self.app.run(**kwargs)

    def list_documents(self):
        documents = [{"name": f, "url": f"/files/{f}", "extension": os.path.splitext(f)[1][1:]} for f in self.files]
        return jsonify(documents)

    def serve_file(self, filename):
        return send_from_directory(self.app.config['UPLOAD_FOLDER'], filename)

    def index(self):
        return render_template('local.html')

    def get_Chat_response(self, query):
        #TODO ajouter un attribut ragchain
        inputs = {
            "query": str(query),
            "chat_history": []
        }
        res = rag_conv._call(inputs)

        output = jsonify({
            'response': res['result'],
            'context': res['context'],
            'source': res['source']
        })
        return output

    def update_settings(self):
        data = request.get_json()
        #TODO
        load_rag(settings=data)
        return jsonify({'status': 'success', 'message': 'Settings updated successfully'}), 200

    def clear_chat_history(self):
        #TODO
        rag_conv.clear_chat_history()
        return jsonify({'status': 'success', 'message': 'Chat history cleared'}), 200