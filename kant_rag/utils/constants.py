from ragas.langchain import RagasEvaluatorChain
from ragas.metrics import (
    answer_relevancy,
    context_recall,
    context_relevancy,
    faithfulness,
)

EMBEDDINGS_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
EMBEDDINGS_KWARGS = {"device": "cuda"}
ENCODE_KWARGS = {"normalize_embeddings": False}
K_VECTORS = 5
OPENAI_KEY = "..."
OPENAI_MODEL = "text-davinci-003"
PROMPT_INPUT_VARIABLES = ["context", "question"]
EVALUATION_METRICS = [
    "faithfulness_score",
    "answer_relevancy_score",
    "context_relevancy_score",
    "context_recall_score",
]
EVALUATORS = [
    RagasEvaluatorChain(metric=faithfulness),
    RagasEvaluatorChain(metric=answer_relevancy),
    RagasEvaluatorChain(metric=context_relevancy),
    RagasEvaluatorChain(metric=context_recall),
]
EVALUATION_ZIP = zip(EVALUATION_METRICS, EVALUATORS)
TEMPERATURE = 0.6
CHAIN_TYPE = "stuff"
DEFAULT_QUESTION = "Do I have a moral obligation to do the right thing?"
DEFAULT_SCAN = ["hallucination"]
TEMPLATE = """
    Don't try to make up an answer, if you don't know just say that you don't know.
    Answer in the same language the question was asked.
    Use only the following pieces of context to answer the question at the end.

    {context}

    Question: {question}
    Answer:"""


ragas_dict = dict()
for key, evaluator in EVALUATION_ZIP:
    metric = [
        evaluator.evaluate(x["examples", x["result"]]) for x in prepared_responses
    ]
    if metric != list():
        metric = [x[key] for x in metric]
    else:
        metric = None
    ragas_dict[key] = metric
