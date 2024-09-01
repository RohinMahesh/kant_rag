import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import tiktoken
from kant_rag.base.objects import ResponseValidator
from kant_rag.estimator.confidence_estimator import EstimateConfidence
from kant_rag.utils.constants import (
    CHAIN_TYPE,
    CONTEXT_WINDOW,
    DEFAULT_QUESTION,
    K_VECTORS,
    OPENAI_KEY,
    OPENAI_MODEL,
    PRINCIPLES,
    PROMPT_INPUT_VARIABLES,
    SEED,
    TEMPERATURE,
    TEMPLATE,
    TIKTOKEN_ENCODING,
)
from kant_rag.utils.file_paths import INDEX_PATH
from kant_rag.utils.helpers import count_tokens, load_embeddings
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains.constitutional_ai.base import ConstitutionalChain
from langchain.chains.llm import LLMChain
from langchain.schema import HumanMessage
from langchain.vectorstores import FAISS
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_openai import ChatOpenAI


@dataclass
class KantRAG:
    """
    Creates RetrievalQA chain for QA using LangChain and OpenAI

    :param question: question to answer,
        defaults to DEFAULT_QUESTION
    :param temperature: temperature for Open AI model,
        defaults to TEMPERATURE
    :param number_similar_embedding: number of similar embeddings to search via FAISS,
        defaults to K_VECTORS
    :param debug: boolean value to return RAG model for testing,
        defaults to FALSE
    :param embedding_type: type of embeddings to use for searching and asking,
        defaults to 'HuggingFace'
    :param parser: PyDantic output parser for formatting output response,
        defaults to ResponseValidator
    """

    question: str = DEFAULT_QUESTION
    temperature: float = TEMPERATURE
    number_similar_vectors: int = K_VECTORS
    debug: bool = False
    embedding_type: str = "HuggingFace"
    parser: PydanticOutputParser = PydanticOutputParser(
        pydantic_object=ResponseValidator
    )

    def _process_output(self, text: Dict[str, Any]) -> Dict[str, Union[str, List[str]]]:
        """
        Formats response from OpenAI

        :param text: response from OpenAI
        :returns dictionary containing response and sources
        """
        # Clean result
        response = text["result"].strip()

        # Extract source documents
        source_documents = [
            x.metadata["page"]["Source"] for x in text["source_documents"]
        ]
        return {
            "response": response,
            "source": source_documents,
            "confidence": text["confidence"],
        }

    def _rewrite_response(self) -> str:
        """
        Applies LangChain Constitutional Chain to rewrite responses that violates principles

        :returns rewritten response
        """
        # Get principles to be applied
        principles = ConstitutionalChain.get_principles(PRINCIPLES)

        # Define constitutional chain
        constitutional_chain = ConstitutionalChain.from_llm(
            chain=LLMChain(llm=self.llm, prompt=self.prompt),
            constitutional_principles=principles,
            llm=self.llm,
            verbose=True,
        )

        # Rewrite response for violations of principles
        response = constitutional_chain.run(
            question=self.question, context=self.context
        )
        return response

    def _create_chain(self) -> None:
        """
        Creates LangChain RetrievalQA object

        :returns None
        """
        # Load embeddings
        embeddings = load_embeddings(embedding_type=self.embedding_type)

        # Load FAISS index
        faiss_db = FAISS.load_local(INDEX_PATH, embeddings)

        # Get similar documents from knowledge base for context
        similar_documents = faiss_db.similarity_search(self.question, k=K_VECTORS)

        # Retrieve and select the most relevant context
        context = "\n".join([doc.page_content for doc in similar_documents])

        # Ensure the context fits within the context window size
        token_count = count_tokens(context)
        if token_count > CONTEXT_WINDOW:
            warnings.warn(
                f"For model {OPENAI_MODEL}, the selected context exceeds the context window size of {CONTEXT_WINDOW} tokens. Trimming the context."
            )
            context_tokens = tiktoken.get_encoding(TIKTOKEN_ENCODING).encode(context)
            context = tiktoken.get_encoding(TIKTOKEN_ENCODING).decode(
                context_tokens[:CONTEXT_WINDOW]
            )

        self.context = context

        # Define LangChain prompt template
        self.prompt = PromptTemplate(
            template=TEMPLATE,
            input_variables=PROMPT_INPUT_VARIABLES,
            partial_variables={
                "formatting_instructions": self.parser.get_format_instructions()
            },
        )

        # Initialize OpenAI client
        self.llm = ChatOpenAI(
            model=OPENAI_MODEL,
            temperature=self.temperature,
            seed=SEED,
            api_key=OPENAI_KEY,
        )

        # Initialize retriever to get similar vectors from FAISS
        retriever = faiss_db.as_retriever(
            search_kwargs={
                "k": self.number_similar_vectors,
                "search_type": "similarity",
            }
        )

        # Define Langchain RetrievalQA Chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type=CHAIN_TYPE,
            retriever=retriever,
            chain_type_kwargs={"prompt": self.prompt},
            return_source_documents=True,
            verbose=False,
        )

        # Get confidence estimation from LLM
        formatted_context = TEMPLATE.format(
            context=self.context, question=self.question
        )
        human_message = HumanMessage(content=formatted_context)
        message = self.llm.invoke(input=[human_message])
        logprobs = [
            token_info["logprob"]
            for token_info in message.response_metadata["logprobs"]["content"]
        ]
        self.confidence_metrics = EstimateConfidence(logprobs=logprobs).estimate()

    def run(self) -> Union[Dict[str, Union[str, List[str]]], RetrievalQA]:
        """
        Constructs context and prompt to answer given question

        :returns response from OpenAI given constructed prompt
        """
        # Create RetrievalQA chain
        self._create_chain()

        if self.debug:
            return self.qa_chain

        # Get response from OpenAI
        text = self.qa_chain({"context": self.context, "query": self.question})
        text.update({"confidence": self.confidence_metrics})

        # Validate response with PyDantic
        validated_response = ResponseValidator(**text)

        # Process response
        formatted_response = self._process_output(text)

        # Validate no content violations
        formatted_response["response"] = self._rewrite_response()

        return formatted_response
