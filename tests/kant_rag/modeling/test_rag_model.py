from unittest.mock import MagicMock, patch

import pytest
from kant_rag.modeling.rag_model import KantRAG
from langchain.schema import Document
from langchain_openai import ChatOpenAI


@pytest.mark.parametrize(
    "text, expected_output",
    [
        (
            {
                "context": "....",
                "query": "Do I have a moral obligation to do the right thing?",
                "result": "Yes, according to the context provided, you have a moral obligation to do the right thing. This obligation arises from the recognition of laws that govern moral actions, which are determined by practical reason. Even if your inclinations do not align with these laws, the duty to act according to them remains.",
                "source_documents": [
                    Document(
                        page_content="...",
                        metadata={"page": {"Source": "Section 1 Paragraph 10"}},
                        _lc_kwargs={
                            "page_content": "...",
                            "metadata": {"page": {"Source": "Section 1 Paragraph 10"}},
                        },
                    ),
                    Document(
                        page_content="...",
                        metadata={"page": {"Source": "Section 1 Paragraph 18"}},
                        _lc_kwargs={
                            "page_content": "...",
                            "metadata": {"page": {"Source": "Section 1 Paragraph 18"}},
                        },
                    ),
                    Document(
                        page_content="...",
                        metadata={"page": {"Source": "Section 3 Paragraph 22"}},
                        _lc_kwargs={
                            "page_content": "...",
                            "metadata": {"page": {"Source": "Section 3 Paragraph 22"}},
                        },
                    ),
                    Document(
                        page_content="...",
                        metadata={"page": {"Source": "Section 2 Paragraph 21"}},
                        _lc_kwargs={
                            "page_content": "...",
                            "metadata": {"page": {"Source": "Section 2 Paragraph 21"}},
                        },
                    ),
                    Document(
                        page_content="...",
                        metadata={"page": {"Source": "Section 2 Paragraph 14"}},
                        _lc_kwargs={
                            "page_content": "...",
                            "metadata": {"page": {"Source": "Section 2 Paragraph 14"}},
                        },
                    ),
                ],
            },
            {
                "response": "Yes, according to the context provided, you have a moral obligation to do the right thing. This obligation arises from the recognition of laws that govern moral actions, which are determined by practical reason. Even if your inclinations do not align with these laws, the duty to act according to them remains.",
                "source": [
                    "Section 1 Paragraph 10",
                    "Section 1 Paragraph 18",
                    "Section 3 Paragraph 22",
                    "Section 2 Paragraph 21",
                    "Section 2 Paragraph 14",
                ],
            },
        ),
        (
            {
                "context": "...",
                "query": "Do some actions have a higher moral worth than others?",
                "result": "Yes, some actions can be considered to have a higher moral worth than others, particularly when they are performed out of a sense of duty and respect for the moral law, rather than for personal gain or external rewards. Actions that align with the principle of treating others as ends in themselves, rather than merely as means to an end, are seen as possessing greater moral significance.",
                "source_documents": [
                    Document(
                        page_content="...",
                        metadata={"page": {"Source": "Section 2 Paragraph 92"}},
                        _lc_kwargs={
                            "page_content": "...",
                            "metadata": {"page": {"Source": "Section 2 Paragraph 92"}},
                        },
                    ),
                    Document(
                        page_content="...",
                        metadata={"page": {"Source": "Section 1 Paragraph 6"}},
                        _lc_kwargs={
                            "page_content": "...",
                            "metadata": {"page": {"Source": "Section 1 Paragraph 6"}},
                        },
                    ),
                    Document(
                        page_content="...",
                        metadata={"page": {"Source": "Section 3 Paragraph 8"}},
                        _lc_kwargs={
                            "page_content": "...",
                            "metadata": {"page": {"Source": "Section 3 Paragraph 8"}},
                        },
                    ),
                    Document(
                        page_content="...",
                        metadata={"page": {"Source": "Section 3 Paragraph 9"}},
                        _lc_kwargs={
                            "page_content": "...",
                            "metadata": {"page": {"Source": "Section 3 Paragraph 9"}},
                        },
                    ),
                    Document(
                        page_content="...",
                        metadata={"page": {"Source": "Section 2 Paragraph 60"}},
                        _lc_kwargs={
                            "page_content": "...",
                            "metadata": {"page": {"Source": "Section 2 Paragraph 60"}},
                        },
                    ),
                ],
            },
            {
                "response": "Yes, some actions can be considered to have a higher moral worth than others, particularly when they are performed out of a sense of duty and respect for the moral law, rather than for personal gain or external rewards. Actions that align with the principle of treating others as ends in themselves, rather than merely as means to an end, are seen as possessing greater moral significance.",
                "source": [
                    "Section 2 Paragraph 92",
                    "Section 1 Paragraph 6",
                    "Section 3 Paragraph 8",
                    "Section 3 Paragraph 9",
                    "Section 2 Paragraph 60",
                ],
            },
        ),
    ],
)
def test_process_output(text, expected_output):
    kant_rag = KantRAG()
    output = kant_rag._process_output(text=text)
    assert output == expected_output


@pytest.mark.parametrize(
    "query, text, expected_output",
    [
        (
            "Do I have a moral obligation to do the right thing?",
            {
                "context": "....",
                "query": "Do I have a moral obligation to do the right thing?",
                "result": "Yes, according to the context provided, you have a moral obligation to do the right thing. This obligation arises from the recognition of laws that govern moral actions, which are determined by practical reason. Even if your inclinations do not align with these laws, the duty to act according to them remains.",
                "source_documents": [
                    Document(
                        page_content="...",
                        metadata={"page": {"Source": "Section 1 Paragraph 10"}},
                        _lc_kwargs={
                            "page_content": "...",
                            "metadata": {"page": {"Source": "Section 1 Paragraph 10"}},
                        },
                    ),
                    Document(
                        page_content="...",
                        metadata={"page": {"Source": "Section 1 Paragraph 18"}},
                        _lc_kwargs={
                            "page_content": "...",
                            "metadata": {"page": {"Source": "Section 1 Paragraph 18"}},
                        },
                    ),
                    Document(
                        page_content="...",
                        metadata={"page": {"Source": "Section 3 Paragraph 22"}},
                        _lc_kwargs={
                            "page_content": "...",
                            "metadata": {"page": {"Source": "Section 3 Paragraph 22"}},
                        },
                    ),
                    Document(
                        page_content="...",
                        metadata={"page": {"Source": "Section 2 Paragraph 21"}},
                        _lc_kwargs={
                            "page_content": "...",
                            "metadata": {"page": {"Source": "Section 2 Paragraph 21"}},
                        },
                    ),
                    Document(
                        page_content="...",
                        metadata={"page": {"Source": "Section 2 Paragraph 14"}},
                        _lc_kwargs={
                            "page_content": "...",
                            "metadata": {"page": {"Source": "Section 2 Paragraph 14"}},
                        },
                    ),
                ],
            },
            {
                "response": "Yes, according to the context provided, you have a moral obligation to do the right thing. This obligation arises from the recognition of laws that govern moral actions, which are determined by practical reason. Even if your inclinations do not align with these laws, the duty to act according to them remains.",
                "source": [
                    "Section 1 Paragraph 10",
                    "Section 1 Paragraph 18",
                    "Section 3 Paragraph 22",
                    "Section 2 Paragraph 21",
                    "Section 2 Paragraph 14",
                ],
            },
        ),
        (
            "Do some actions have a higher moral worth than others?",
            {
                "context": "...",
                "query": "Do some actions have a higher moral worth than others?",
                "result": "Yes, some actions can be considered to have a higher moral worth than others, particularly when they are performed out of a sense of duty and respect for the moral law, rather than for personal gain or external rewards. Actions that align with the principle of treating others as ends in themselves, rather than merely as means to an end, are seen as possessing greater moral significance.",
                "source_documents": [
                    Document(
                        page_content="...",
                        metadata={"page": {"Source": "Section 2 Paragraph 92"}},
                        _lc_kwargs={
                            "page_content": "...",
                            "metadata": {"page": {"Source": "Section 2 Paragraph 92"}},
                        },
                    ),
                    Document(
                        page_content="...",
                        metadata={"page": {"Source": "Section 1 Paragraph 6"}},
                        _lc_kwargs={
                            "page_content": "...",
                            "metadata": {"page": {"Source": "Section 1 Paragraph 6"}},
                        },
                    ),
                    Document(
                        page_content="...",
                        metadata={"page": {"Source": "Section 3 Paragraph 8"}},
                        _lc_kwargs={
                            "page_content": "...",
                            "metadata": {"page": {"Source": "Section 3 Paragraph 8"}},
                        },
                    ),
                    Document(
                        page_content="...",
                        metadata={"page": {"Source": "Section 3 Paragraph 9"}},
                        _lc_kwargs={
                            "page_content": "...",
                            "metadata": {"page": {"Source": "Section 3 Paragraph 9"}},
                        },
                    ),
                    Document(
                        page_content="...",
                        metadata={"page": {"Source": "Section 2 Paragraph 60"}},
                        _lc_kwargs={
                            "page_content": "...",
                            "metadata": {"page": {"Source": "Section 2 Paragraph 60"}},
                        },
                    ),
                ],
            },
            {
                "response": "Yes, some actions can be considered to have a higher moral worth than others, particularly when they are performed out of a sense of duty and respect for the moral law, rather than for personal gain or external rewards. Actions that align with the principle of treating others as ends in themselves, rather than merely as means to an end, are seen as possessing greater moral significance.",
                "source": [
                    "Section 2 Paragraph 92",
                    "Section 1 Paragraph 6",
                    "Section 3 Paragraph 8",
                    "Section 3 Paragraph 9",
                    "Section 2 Paragraph 60",
                ],
            },
        ),
    ],
)
@patch("kant_rag.modeling.rag_model.KantRAG._create_chain")
@patch.object(ChatOpenAI, "__call__")
def test_run(mock_chat_openai_call, mock_create_chain, query, text, expected_output):
    # Set up a mock for the qa_chain attribute
    mock_qa_chain = MagicMock()
    mock_qa_chain.return_value = text

    # Ensure that the mocked _create_chain method sets the qa_chain, context, and query attributes
    def side_effect_create_chain():
        setattr(kant_rag, "qa_chain", mock_qa_chain)
        setattr(kant_rag, "context", text["context"])
        setattr(kant_rag, "question", text["query"])

    mock_create_chain.side_effect = side_effect_create_chain

    # Instantiate and run KantRAG
    kant_rag = KantRAG(question=query)
    response = kant_rag.run()

    # Assertion for output with PyDantic validation
    assert response == expected_output
