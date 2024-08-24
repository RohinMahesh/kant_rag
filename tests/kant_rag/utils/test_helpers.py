import os
import sys

repo_path = os.path.abspath("/Users/rohinmahesh/Documents/GitHub/kant_rag")

if repo_path not in sys.path:
    sys.path.append(repo_path)

import logging
from pathlib import Path
import pytest
import shutil
import tempfile
from unittest.mock import patch, MagicMock

from kant_rag.utils.helpers import count_tokens, load_embeddings, create_save_faiss_db
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings


# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "input_text, expected_output",
    [
        ("Hello world", 2),
        ("", 0),
    ],
)
def test_count_tokens(input_text, expected_output):
    assert count_tokens(text=input_text) == expected_output


@pytest.mark.parametrize(
    "embedding_type, expected_class",
    [
        ("HuggingFace", HuggingFaceEmbeddings),
        ("OpenAI", OpenAIEmbeddings),
        ("SomethingElse", OpenAIEmbeddings),
    ],
)
@patch("kant_rag.utils.helpers.HuggingFaceEmbeddings")
@patch("kant_rag.utils.helpers.OpenAIEmbeddings")
def test_load_embeddings(
    mock_openai_embeddings, mock_hf_embeddings, embedding_type, expected_class
):
    # Assign mock instances
    mock_hf_embeddings.return_value = MagicMock(spec=HuggingFaceEmbeddings)
    mock_openai_embeddings.return_value = MagicMock(spec=OpenAIEmbeddings)
    embeddings = load_embeddings(embedding_type=embedding_type)

    # Verify the correct class was instantiated
    assert isinstance(embeddings, expected_class)


@pytest.mark.parametrize("embedding_type", ["HuggingFace", "OpenAI"])
@patch("kant_rag.utils.helpers.load_embeddings")
@patch("kant_rag.utils.helpers.FAISS")
def test_create_save_faiss_db(mock_faiss, mock_load_embeddings, embedding_type):
    # Create a temporary directory for the test
    temp_dir = tempfile.mkdtemp()
    temp_dir_path = Path(temp_dir)

    # Log the temporary directory path
    logger.debug(f"Temporary directory for test: {temp_dir_path}")

    # Mocking the embeddings method and FAISS operations
    mock_embeddings = MagicMock()
    mock_load_embeddings.return_value = mock_embeddings

    # Mock the FAISS from_documents method to simulate index creation
    mock_db = MagicMock()
    mock_faiss.from_documents.return_value = mock_db

    # Mock the save_local method to simulate file creation
    def mock_save_local(path):
        (Path(path) / "index.faiss").touch()
        (Path(path) / "index.pkl").touch()

    mock_db.save_local.side_effect = mock_save_local

    # Example input data
    text = ["Document 1", "Document 2"]
    metadata = ["Source 1", "Source 2"]

    # Call the function with the temporary directory as the storage path
    create_save_faiss_db(
        text=text,
        metadata=metadata,
        embedding_type=embedding_type,
        storage_dir=str(temp_dir_path),
    )

    # Check if the index files are created in the temporary directory
    faiss_file = temp_dir_path / "index.faiss"
    pkl_file = temp_dir_path / "index.pkl"

    # Assertions
    assert faiss_file.exists(), "index.faiss file was not created."
    assert pkl_file.exists(), "index.pkl file was not created."

    # Clean up the temporary directory after the test
    shutil.rmtree(temp_dir)
