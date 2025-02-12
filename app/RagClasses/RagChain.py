import os
import sys

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../python_script')))

from app.RagClasses import CustomEmbeddings
from app.RagClasses import AppConfig
from app.RagClasses import Database
from app.RagClasses import LLM


class RagChain:
    def __init__(self, config):
        self.config = AppConfig(config=config).as_dict()
        self.embedding_model = CustomEmbeddings(model_name=self.config["embedding_model"]).model
        #TODO
        llm = LLM(model_name=self.config["llm_model"])
        self.llm_model = llm.model
        self.database = Database(config=self.config).database
        self.retriever = self.load_retriever()
        self.rag_chain = self.load_rag_chain()

    def load_retriever(self):
        search_type = self.config["search_type"]
        if search_type == "similarity":
            retriever = self.database.as_retriever(search_type=search_type,
                                                   search_kwargs={"k": self.config["similarity_doc_nb"]})

        elif search_type == "similarity_score_threshold":
            retriever = self.database.as_retriever(search_type=search_type,
                                                   search_kwargs={"k": self.config["max_chunk_return"],
                                                                  "score_threshold": self.config[
                                                                      "score_threshold"]})

        elif search_type == "mmr":
            retriever = self.database.as_retriever(search_type=search_type,
                                                   search_kwargs={"k": self.config["mmr_doc_nb"],
                                                                  "fetch_k": self.config["considered_chunk"],
                                                                  "lambda_mult": self.config["lambda_mult"]})

        else:
            raise ValueError("Invalid 'search_type' setting")

        if self.config["isHistoryOn"]:
            contextualize_q_system_prompt = """Étant donné un historique de chat et la dernière question de l'utilisateur, qui pourrait faire référence à un contexte dans l'historique, reformulez la question de manière autonome, de sorte qu'elle soit compréhensible sans l'historique de chat. Ne répondez pas à la question, contentez-vous de la reformuler si nécessaire, sinon renvoyez-la telle quelle."""

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
        qa_system_prompt = """Tu es un assistant, membre de l'école d'ingénieur Seatech, pour des tâches de question-réponse. Utilise les éléments suivants du contexte récupéré pour répondre à la question. Si tu ne connais pas la réponse, dis simplement que tu ne sais pas.

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
        old_parameters = self.config.copy()
        self.config.update(new_parameters)
        need_reload_db = False
        need_reload_retriever = False

        if old_parameters["embedding_model"] != self.config["embedding_model"]:
            self.embedding_model = CustomEmbeddings(model_name=self.config["embedding_model"]).model
            need_reload_db = True
            need_reload_retriever = True

        if old_parameters["chroma_root_path"] != self.config["chroma_root_path"]:
            need_reload_db = True
            need_reload_retriever = True

        if not need_reload_db:
            if any(old_parameters[key] != self.config[key] for key in [
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