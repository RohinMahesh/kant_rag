import os
from dataclasses import dataclass
from typing import Dict

from datasets import Dataset
from kant_rag.utils.constants import OPENAI_KEY, RAGAS_MANDATORY_KEYS

os.environ["OPENAI_API_KEY"] = OPENAI_KEY

from ragas import evaluate
from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    context_entity_recall,
    context_precision,
    context_recall,
    context_utilization,
    faithfulness,
)


@dataclass
class EvaluateRAG:
    """
    Evaluates RAG using Ragas

    :param data: dictionary containing queries, answers, contexts, and ground truths
    """

    data: Dict

    def __post_init__(self):
        assert all(
            item in self.data for item in RAGAS_MANDATORY_KEYS
        ), "One of the following mandatory fields of 'question', 'answer', 'contexts', and 'ground_truth' is missing!"

        object.__setattr__(
            self,
            "evaluators",
            [
                faithfulness,
                answer_relevancy,
                answer_correctness,
                context_recall,
                context_precision,
                context_utilization,
                context_entity_recall,
            ],
        )

    def score_rag(self) -> Dict[str, float]:
        """
        Scores RAG for generation and retrieval

        :returns ragas_metrics: evaluation metrics for generation/retreival of RAG
        """
        # Prepare data
        data = Dataset.from_dict(self.data)

        # Evaluate performance
        scores = evaluate(data, metrics=self.evaluators)
        scores = {key: round(value, 4) for key, value in scores.items()}
        return scores
