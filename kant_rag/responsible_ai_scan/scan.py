from dataclasses import dataclass
from typing import List, Union

import giskard
import openai
from kant_rag.modeling.rag_model import KantRAG
from kant_rag.utils.constants import DEFAULT_SCAN, OPENAI_KEY


@dataclass
class ScanRAGModel:
    """
    Performs scans for various risks in RAG model to promote Responsible AI, including:
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
    scans: List[str] = DEFAULT_SCAN
    interactive: bool = False

    def evaluate(self) -> Union[dict, giskard.Report]:
        # Get RAG model for evaluation
        openai.api_key = OPENAI_KEY
        rag_model = KantRAG(debug=True).run()

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
