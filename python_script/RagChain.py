import os
import sys

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../python_script')))
from Embedding import Embeddings
from LLM import LLM
from populate_database import find_chroma_path


class RagChain:
    def __init__(self, parameters: dict):
        self.parameters: dict = self.load_parametres(parameters)
        self.embedding_model: Embeddings = Embeddings(model_name=self.parameters["embedding_model"]).model
        self.llm_model = LLM(model_name=self.parameters["llm_model"]).llm_model
        self.database = self.load_database()
        self.retriever = self.load_retriever()
        self.rag_chain = self.load_rag_chain()

    @staticmethod
    def load_parametres(parameters):
        default_params = {
            "chroma_root_path": "data/test/embedded_database",
            "embedding_model": "voyage-3",
            "llm_model": "gpt-3.5-turbo",
            "search_type": "similarity",
            "similarity_doc_nb": 5,
            "score_threshold": 0.8,
            "max_chunk_return": 5,
            "considered_chunk": 25,
            "mmr_doc_nb": 5,
            "lambda_mult": 0.25,
            "isHistoryOn": True,
        }

        if parameters is None:
            print("Default parameters used:", default_params)
            return default_params
        else:
            merged_params = {**default_params, **parameters}
            default_used_params = {key: value for key, value in default_params.items() if key not in parameters}

            if default_used_params:
                print("Parameters that remain as default:", default_used_params)

            return merged_params

    def load_database(self):
        db = Chroma(persist_directory=find_chroma_path(model_name=self.parameters["embedding_model"],
                                                       base_path=self.parameters["chroma_root_path"]),
                    embedding_function=self.embedding_model)
        return db

    def load_retriever(self):
        search_type = self.parameters["search_type"]
        if search_type == "similarity":
            retriever = self.database.as_retriever(search_type=search_type,
                                                   search_kwargs={"k": self.parameters["similarity_doc_nb"]})

        elif search_type == "similarity_score_threshold":
            retriever = self.database.as_retriever(search_type=search_type,
                                                   search_kwargs={"k": self.parameters["max_chunk_return"],
                                                                  "score_threshold": self.parameters[
                                                                      "score_threshold"]})

        elif search_type == "mmr":
            retriever = self.database.as_retriever(search_type=search_type,
                                                   search_kwargs={"k": self.parameters["mmr_doc_nb"],
                                                                  "fetch_k": self.parameters["considered_chunk"],
                                                                  "lambda_mult": self.parameters["lambda_mult"]})

        else:
            raise ValueError("Invalid 'search_type' setting")

        if self.parameters["isHistoryOn"]:
            contextualize_q_system_prompt = """Given a chat history and the latest user question \
            which might reference context in the chat history, formulate a standalone question \
            which can be understood without the chat history. Do NOT answer the question, \
            just reformulate it if needed and otherwise return it as is."""

            contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", contextualize_q_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )

            history_aware_retriever = create_history_aware_retriever(
                self.llm_model, retriever, contextualize_q_prompt
            )
            retriever = history_aware_retriever
        return retriever

    def load_rag_chain(self):
        qa_system_prompt = """You are an assistant for question-answering tasks. \
        Use the following pieces of retrieved context to answer the question. \
        If you don't know the answer, just say that you don't know. \
        Use three sentences maximum and keep the answer concise.\

        {context}"""
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(self.llm_model, qa_prompt)
        return create_retrieval_chain(self.retriever, question_answer_chain)

    def update_parameters(self, new_parameters):
        old_parameters = self.parameters.copy()
        self.parameters.update(new_parameters)
        need_reload_db = False
        need_reload_retriever = False

        if old_parameters["embedding_model"] != self.parameters["embedding_model"]:
            self.embedding_model = Embeddings(model_name=self.parameters["embedding_model"]).model
            need_reload_db = True
            need_reload_retriever = True

        if old_parameters["chroma_root_path"] != self.parameters["chroma_root_path"]:
            need_reload_db = True
            need_reload_retriever = True

        if not need_reload_db:
            if any(old_parameters[key] != self.parameters[key] for key in [
                "search_type", "similarity_doc_nb", "score_threshold",
                "max_chunk_return", "considered_chunk", "mmr_doc_nb",
                "isHistoryOn", "lambda_mult"
            ]):
                need_reload_retriever = True

        if need_reload_db:
            self.load_database()

        if need_reload_retriever:
            self.load_retriever()

        self.rag_chain = self.load_rag_chain()
