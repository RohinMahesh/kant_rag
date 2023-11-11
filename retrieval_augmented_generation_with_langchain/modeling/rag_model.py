from dataclasses import dataclass

from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS

from retrieval_augmented_generation_with_langchain.utils.constants import (
    CHAIN_TYPE,
    K_VECTORS,
    OPENAI_KEY,
    OPENAI_MODEL,
    PROMPT_INPUT_VARIABLES,
    TEMPERATURE,
    TEMPLATE,
)

from retrieval_augmented_generation_with_langchain.utils.file_paths import INDEX_PATH
from retrieval_augmented_generation_with_langchain.utils.helpers import load_embeddings


@dataclass
class PhilosophyRAG:
    """
    Performs Retrieval Augmented Generation with Open AI and LangChain

    :param question: question to answer
    :param temperature: temperature for Open AI model,
        defaults to TEMPERATURE
    :param number_similar_embedding: number of similar embeddings to search via FAISS,
        defaults to K_VECTORS
    """

    question: str
    temperate: float = TEMPERATURE
    number_similar_vectors: int = K_VECTORS

    def _process_output(self, text: str):
        """
        Formats response from OpenAI

        :param text: response from OpenAI
        :returns dictionary containing response and sources
        """
        response = text["result"].strip()
        source_documents = [x for x in text["metadata"]["page"]["Source"]]
        return {"Response": response, "Source": source_documents}

    def run(self):
        """
        Constructs context and prompt to answer given question

        :returns response from OpenAI given constructed prompt
        """
        # Load embeddings
        embeddings = load_embeddings()

        # Load FAISS index
        faiss_db = FAISS.load_local(INDEX_PATH, embeddings)

        # Define LangChain prompt template
        prompt = PromptTemplate(
            template=TEMPLATE, input_variables=PROMPT_INPUT_VARIABLES
        )

        # Initialize OpenAI client
        llm = OpenAI(
            openai_api_key=OPENAI_KEY, model_name=OPENAI_MODEL, temperature=TEMPERATURE
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
            llm=llm,
            chain_type=CHAIN_TYPE,
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True,
            verbose=False,
        )

        # Get response from OpenAI
        text = self.qa_chain(self.question)

        # Process response
        formatted_response = self._process_output(text)

        # Get response from OpenAI
        return formatted_response
