from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
from langchain.chains import RetrievalQA

from kant_rag.modeling.rag_model import KantRAG
from kant_rag.utils.constants import EVALUATION_ZIP


@dataclass
class EvaluateRAG:
    """
    Evaluates RAG using Ragas

    :param queries: list of queries for evaluation
    :param responses: responses from RAG
    :param rag_model: LangChain RetrievalQA chain,
        defaults to KantRAG
    :param eval_zip: evaluation zip containing metric and RagasEvaluatorChain,
        defaults to EVALUATION_ZIP
    """

    queries: List[str]
    responses: List[str]
    rag_model: RetrievalQA = KantRAG
    eval_zip: zip = EVALUATION_ZIP

    def _prepare_responses(self) -> Dict[str, str]:
        """
        Prepares responses for evaluation

        :return prepared_responses: prepared examples andresponses from RetrievalQA chain
        """
        # Prepare input for batch RAG model
        examples = [
            {"query": q, "ground_truths": [self.responses[i]]}
            for i, q in enumerate(self.queries)
        ]

        # Get batch responses
        result = self.rag_model.batch(examples)

        prepared_responses = {
            "examples": examples,
            "result": result,
        }
        return prepared_responses

    def score_rag(self) -> Dict[str, float]:
        """
        Scores RAG for generation and retrieval

        :returns ragas_metrics: evaluation metrics for generation/retreival of RAG
        """
        # Get RAG response
        prepared_response = self._prepare_examples()

        # Calculate metrics for batch of examples
        ragas_metrics = dict()
        for key, evaluator in EVALUATION_ZIP:
            metric = [
                evaluator.evaluate(x["examples", x["result"]])
                for x in prepared_response
            ]
            if metric != list():
                metric = np.sum([x[key] for x in metric])
                ragas_metrics[key] = round(metric, 2)

        # Calculate ragas score
        ragas_metrics["ragas_score"] = round(
            np.mean([np.sum(x) for x in list(ragas_metrics.values())]), 2
        )
        return ragas_metrics
