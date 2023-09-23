EMBEDDINGS_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
EMBEDDINGS_KWARGS = {"device": "cuda"}
ENCODE_KWARGS = {"normalize_embeddings": False}
K_VECTORS = 5
OPENAI_KEY = "..."
PROMPT_INPUT_VARIABLES = ["context", "question"]
TEMPERATURE = 0.6
TEMPLATE = """
    Don't try to make up an answer, if you don't know just say that you don't know.
    Answer in the same language the question was asked.
    Use only the following pieces of context to answer the question at the end.

    {context}

    Question: {question}
    Answer:"""
