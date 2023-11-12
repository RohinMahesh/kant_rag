from dataclasses import dataclass

import giskard
import openai
from retrieval_augmented_generation_with_langchain.modeling.rag_model import (
    PhilosophyRAG,
)
from retrieval_augmented_generation_with_langchain.utils.constants import (
    DEFAULT_QUESTION,
    DEFAULT_SCAN,
    OPENAI_KEY,
)


@dataclass
class ScanRAGModel:
    """
    Performs scans for various risks in RAG model, including:
        - Injection attacks
        - Hallucination and misinformation
        - Harmful content generation
        - Stereotypes
        - Information disclosure

    :param name: name of giskard model
    :param description: description of giskard model
    :param scans: risk scans to perform,
        defaults to DEFAULT_SCAN
    :param interactive: indicator for interactive report,
        defaults to False
    """

    name: str
    description: str
    scans: list = DEFAULT_SCAN
    interactive: bool = False

    def evaluate(self):
        # Get RAG model for evaluation
        openai.api_key = OPENAI_KEY
        rag_model = PhilosophyRAG(debug=True).run()

        # Initialize Giskard model
        giskard_model = giskard.Model(
            model=rag_model,
            model_type="text_generation",
            name=self.name,
            description=self.description,
            feature_names=["query"],
        )

        # Scan for risks in RAG model
        report = giskard.scan(giskard_model, only=self.scans)

        # Return report
        if self.interactive:
            return report

        return report.to_dataframe().to_dict()
