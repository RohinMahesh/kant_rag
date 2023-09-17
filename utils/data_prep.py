import re
from dataclasses import dataclass
from typing import List

import numpy
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
from utils.constants import DEVICE, MODEL_CHECKPOINT, TEXT_COLUMN


@dataclass
class HFEmbeddings:
    """
    Creates embeddings from HuggingFace
    :param data: list containing text for feature extraction
    :param tokenizer: HuggingFace tokenizer
    :param model: HuggingFace model
    """

    data: List[str]
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    model: AutoModel = AutoModel.from_pretrained(MODEL_CHECKPOINT).to(DEVICE)

    def create_embeddings(self):
        """
        Tokenizes data and creates embeddings
        :returns feature_space: HuggingFace embedding vector(s)
        """

        # Encode strings and convert tokens to PyTorch tensors
        encoded = self.tokenizer(
            self.data,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        # Place encoded strings on device
        inputs = {
            k: v.to(DEVICE)
            for k, v in encoded.items()
            if k in self.tokenizer.model_input_names
        }

        # Disable auto calc of the gradient to reduce memory usage of calculation
        with torch.no_grad():
            # Get embeddings and extract the last hidden states
            hidden_states = self.model(**inputs).last_hidden_state

        # Extract vectors for [CLS] token
        feature_space = hidden_states[:, 0].cpu().numpy()

        return feature_space
