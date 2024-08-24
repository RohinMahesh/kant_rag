from typing import List

import tiktoken
from kant_rag.utils.constants import (
    HF_EMBEDDINGS_KWARGS,
    HF_EMBEDDINGS_NAME,
    HF_ENCODE_KWARGS,
    OPENAI_EMBEDDINGS_NAME,
    OPENAI_KEY,
    TIKTOKEN_ENCODING,
)
from kant_rag.utils.file_paths import INDEX_PATH
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.schema import Document
from langchain.vectorstores import FAISS


def count_tokens(text: str) -> int:
    """
    Estimates the number of tokens in a given text via tiktoken

    :param text: the text for analysis
    :returns the estimated number of tokens in the text
    """
    enc = tiktoken.get_encoding(TIKTOKEN_ENCODING)
    return len(enc.encode(text))


def load_embeddings(embedding_type: str) -> HuggingFaceEmbeddings | OpenAIEmbeddings:
    """
    Loads embeddings for searching and asking

    :param embedding_type: embeddings type to load
    :returns embeddings
    """
    return (
        HuggingFaceEmbeddings(
            model_name=HF_EMBEDDINGS_NAME,
            model_kwargs=HF_EMBEDDINGS_KWARGS,
            encode_kwargs=HF_ENCODE_KWARGS,
        )
        if embedding_type == "HuggingFace"
        else OpenAIEmbeddings(model_name=OPENAI_EMBEDDINGS_NAME, api_key=OPENAI_KEY)
    )


def create_save_faiss_db(
    text: List[str],
    metadata: List[str],
    embedding_type: str = "HuggingFace",
    storage_dir: str = INDEX_PATH,
) -> None:
    """
    Creates and saves FAISS index

    :param text: text to be encoded and stored in FAISS
    :param metadata: source material for given text
    :param embedding_type: type of embeddings to load,
        defaults to 'HuggingFace'
    :param storage_dir: directory to store index files,
        defaults to INDEX_PATH
    :returns None
    """
    # Load embeddings
    embeddings = load_embeddings(embedding_type=embedding_type)

    # Prepare metadata
    metadata = [{"Source": x} for x in metadata]

    # Prepare documents
    documents = [
        Document(page_content=x, metadata=dict(page=y)) for x, y in zip(text, metadata)
    ]

    # Create FAISS index
    db = FAISS.from_documents(documents, embeddings)

    # Save FAISS index
    db.save_local(storage_dir)
