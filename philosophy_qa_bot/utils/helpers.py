from typing import Dict, List

import numpy as np
import pandas as pd
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from constants import (
    EMBEDDINGS_NAME,
    EMBEDDINGS_KWARGS,
    ENCODE_KWARGS,
)
from file_paths import INDEX_PATH


def load_embeddings():
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


def create_save_faiss_db(text: List[str], metadata: List[str]):
    """
    Creates and saves FAISS index

    :param text: text to be encoded and stored in FAISS
    :param metadata: source material for given text
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
