from typing import Dict, List

from dataprep.hfembeddings import HFEmbeddings
import faiss
import numpy as np
import pandas as pd
import torch
from utils.constants import HIDDEN_SIZE
from utils.file_paths import INDEX_PATH


def create_index(feature_space: np.ndarray):
    """
    Creates FAISS index

    :param feature_space: numpy array containing text for feature extraction
    """
    # Build the index using hidden size of upstream tokenizer
    index = faiss.IndexFlatL2(feature_space.shape[1])

    # Add vectors to index
    index.add(feature_space)

    # Save index
    faiss.write_index(index, INDEX_PATH)


def search_index(
    input_data: List[str],
    index_file: faiss.IndexFlatL2,
    k: int = 1,
    hidden_size: int = HIDDEN_SIZE,
):
    """
    Searches FAISS index

    :param input_data: list containing text for feature extraction
    :param index_file: FAISS index file
    :param k: optional number of similar vectors for search,
        defaults to 1
    :param hidden_size: optional hidden size for tokenizer,
        defaults to HIDDEN_SIZE
    """
    # Get embeddings
    embedding = HFEmbeddings(data=input_data).create_embeddings()

    # Reshape input to match hidden state of upstream tokenizer
    embedding = embedding.reshape(1, hidden_size)

    # Search index
    distances, indices = index_file.search(embedding, k=k)

    return {"Indices": indices.tolist()[0], "Distances": distances.tolist()[0]}
