from langchain.chains.base import Chain
from langchain.output_parsers import YamlOutputParser
from typing import List, Dict, Any, Optional
from langchain.callbacks.manager import CallbackManagerForChainRun, Callbacks
from langchain.llms.base import BaseLanguageModel
from pydantic import BaseModel, Field

from langchain_core.runnables import RunnableBinding


class ResultYAML(BaseModel):
    result: bool


class ConversationalRagChain(Chain):
    """Chain that encpsulate RAG application enabling natural conversations"""
    rag_chain: RunnableBinding
    yaml_output_parser: YamlOutputParser
    llm: BaseLanguageModel

    # input\output parameters
    input_key: str = "query"
    output_key: str = "result"
    context_key: str = "context"
    source_key: str = "source"

    chat_history: List[Dict[str, str]] = Field(default_factory=list)

    def __init__(self, **data):
        """Initialize the ConversationalRagChain with provided data."""
        super().__init__(**data)

    @property
    def input_keys(self) -> List[str]:
        """List of input keys expected by the chain."""
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """List of output keys produced by the chain."""
        return [self.output_key, self.context_key, self.source_key]

    @classmethod
    def from_llm(
            cls,
            rag_chain: RunnableBinding,
            llm: BaseLanguageModel,
            callbacks: Callbacks = None,
            **kwargs: Any,
    ) -> 'ConversationalRagChain':
        """Initialize the chain from a language model."""
        return cls(
            llm=llm,
            rag_chain=rag_chain,
            yaml_output_parser=YamlOutputParser(pydantic_object=ResultYAML),
            callbacks=callbacks,
            **kwargs,
        )

    @staticmethod
    def format_standalone_response(response):
        """Removes the prompt from the generated response"""

        end_marker = "<|endofprompt|>"
        marker_index = response.find(end_marker)
        if marker_index != -1:
            response = response[marker_index + len(end_marker):].strip()
        return response

    @staticmethod
    def format_outputs(output: Dict[str, Any]) -> tuple:
        """
        Remove the prompt from the generated response and regroup contexts and sources.

        Args:
            output (Dict[str, Any]): The output dictionary containing answer and context.

        Returns:
            tuple: A tuple containing the formatted answer, contexts, and sources.
        """
        answer = output["answer"]
        AI_marker = "Assistant: "
        marker_index = answer.find(AI_marker)
        if marker_index != -1:
            answer = answer[marker_index + len(AI_marker):].strip()
        else:
            AI_marker = "AI: "
            marker_index = answer.find(AI_marker)
            if marker_index != -1:
                answer = answer[marker_index + len(AI_marker):].strip()

        documents = output['context']
        contexts = []
        sources = []
        for doc in documents:
            contexts.append(doc.page_content)
            #TODO Update database which has the last metadata storage system
            try:
                sources.append(doc.metadata['file_name'])
            except:
                sources.append(doc.metadata['url'])
        return answer, contexts, sources

    def update_chat_history(self, user_question, bot_response):
        """Update the chat history with the latest interaction."""
        self.chat_history.append({"role": "user", "content": user_question})
        self.chat_history.append({"role": "ai", "content": bot_response})

    def clear_chat_history(self):
        """Clear the chat history"""
        self.chat_history = []

    def _call(self, inputs: Dict[str, Any], run_manager: Optional[CallbackManagerForChainRun] = None) -> Dict[str, Any]:
        """
        Call the chain to process inputs and return a dictionary with answer, context, and source.

        Args:
            inputs (Dict[str, Any]): Inputs to the chain, expected to contain the query.
            run_manager (Optional[CallbackManagerForChainRun]): Manager for callbacks.

        Returns:
            Dict[str, Any]: Dictionary containing the answer, context, and source.
        """
        chat_history = self.chat_history
        question = inputs[self.input_key]

        output = self.rag_chain.invoke({"input": question, "chat_history": chat_history})
        answer, contexts, sources = self.format_outputs(output)

        if not contexts:
            answer = "No context found, try rephrasing your question"

        self.update_chat_history(question, answer)
        return {self.output_key: answer, self.context_key: contexts, self.source_key: sources}
