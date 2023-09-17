import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN_SIZE = 768
MODEL_CHECKPOINT = "distilbert-base-uncased-finetuned-sst-2-english"
OPENAI_KEY = "..."
OPENAI_MODEL = "text-davinci-003"
PROMPT_INPUT_VARIABLES = ["context", "question"]
TEMPERATURE = 0.5
TEMPLATE = """
    Don't try to make up an answer, if you don't know just say that you don't know.
    Answer in the same language the question was asked.
    Use only the following pieces of context to answer the question at the end.

    {context}

    Question: {question}
    Answer:"""
