from dataclasses import dataclass

import faiss
import pandas as pd
from langchain import LLMChain, PromptTemplate
from langchain.llms import OpenAI
from utils.constants import (
    OPENAI_KEY,
    OPENAI_MODEL,
    PROMPT_INPUT_VARIABLES,
    TEMPERATURE,
    TEMPLATE,
)
from utils.file_paths import INDEX_PATH
from utils.helpers import search_index


@dataclass
class PhilosophyQABot:
    question: str
    data: pd.DataFrame
    temperature: float = TEMPERATURE
    number_similar_embedding: int = 5

    def answer(self):
        """
        Constructs context and prompt to answer given question
        :returns response from OpenAI given constructed prompt
        """
        # Load FAISS index
        faiss_index = faiss.read_index(INDEX_PATH)

        # Identify similar embeddings indices
        similar_docs = search_index(
            input_data=[self.question],
            index_file=faiss_index,
            k=self.number_similar_embedding,
        )

        # Extract documents from indices
        context_vector = self.data[self.data["Index"].isin(similar_docs["Indices"])][
            "Value"
        ].tolist()

        # Define context vector
        context_vector = " ".join(context_vector)

        # Create prompt
        prompt = PromptTemplate(
            template=TEMPLATE, input_variables=PROMPT_INPUT_VARIABLES
        )
        # Initialize OpenAI client
        llm = OpenAI(
            openai_api_key=OPENAI_KEY,
            model_name=OPENAI_MODEL,
            temperature=self.temperature,
        )

        # Define Langchain LLMChain
        llm_chain = LLMChain(prompt=prompt, llm=llm)

        # Get response from OpenAI
        return llm_chain.run(context=context_vector, question=self.question)
