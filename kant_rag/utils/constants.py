OPENAI_EMBEDDINGS_NAME = "text-embedding-ada-002"
HF_EMBEDDINGS_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
HF_EMBEDDINGS_KWARGS = {"device": "cpu"}
HF_ENCODE_KWARGS = {"normalize_embeddings": False}
K_VECTORS = 5
OPENAI_KEY = "abc-123"
OPENAI_MODEL = "gpt-4o-mini-2024-07-18"
PROMPT_INPUT_VARIABLES = ["context", "question"]
RAGAS_MANDATORY_KEYS = ["question", "answer", "contexts", "ground_truth"]
TEMPERATURE = 0.0
CHAIN_TYPE = "stuff"
TIKTOKEN_ENCODING = "cl100k_base"
SEED = 28
CONTEXT_WINDOW = 128000
MAX_OUTPUT_TOKENS = 16000
DEFAULT_QUESTION = "Do I have a moral obligation to do the right thing?"
DEFAULT_SCAN = ["hallucination"]
TEMPLATE = """
    You are an Philosophy AI assistant that provides accurate and concise answers.

    Don't try to make up an answer, if you don't know just say that you don't know.
    Answer in the same language the question was asked.
    Use only the following pieces of context to answer the question at the end.
    Return the response in English.

    {context}

    Question: {question}
"""
