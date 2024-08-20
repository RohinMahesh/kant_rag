from typing import List

import tiktoken
from kant_rag.utils.constants import (
    EMBEDDINGS_KWARGS,
    EMBEDDINGS_NAME,
    ENCODE_KWARGS,
    TIKTOKEN_ENCODING,
)
from kant_rag.utils.file_paths import INDEX_PATH
from langchain.embeddings import HuggingFaceEmbeddings
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


def load_embeddings() -> HuggingFaceEmbeddings:
    """
    Loads HuggingFace Embeddings from LangChain

    :returns embeddings: HuggingFace embeddings
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDINGS_NAME,
        model_kwargs=EMBEDDINGS_KWARGS,
        encode_kwargs=ENCODE_KWARGS,
    )
    return embeddings


def create_save_faiss_db(text: List[str], metadata: List[str]) -> None:
    """
    Creates and saves FAISS index

    :param text: text to be encoded and stored in FAISS
    :param metadata: source material for given text
    :returns None
    """
    # Load embeddings
    embeddings = load_embeddings()

    # Prepare metadata
    metadata = [{"Source": x} for x in metadata]

    # Prepare documents
    documents = [
        Document(page_content=x, metadata=dict(page=y)) for x, y in zip(text, metadata)
    ]

    # Create FAISS index
    db = FAISS.from_documents(documents, embeddings)

    # Save FAISS index
    db.save_local(INDEX_PATH)
