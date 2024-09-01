from unittest.mock import MagicMock, patch

import pytest
from kant_rag.modeling.rag_model import KantRAG
from langchain.schema import Document
from langchain_openai import ChatOpenAI


from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from kant_rag.utils.constants import PRINCIPLES


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
                "confidence": {
                    "average_log_probability": -0.2816,
                    "joint_probability_percentage": 0.0,
                    "perplexity": 1.3252,
                },
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
                "confidence": {
                    "average_log_probability": -0.2816,
                    "joint_probability_percentage": 0.0,
                    "perplexity": 1.3252,
                },
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
                "confidence": {
                    "average_log_probability": -0.2245,
                    "joint_probability_percentage": 0.5,
                    "perplexity": 1.1023,
                },
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
                "confidence": {
                    "average_log_probability": -0.2245,
                    "joint_probability_percentage": 0.5,
                    "perplexity": 1.1023,
                },
            },
        ),
    ],
)
def test_process_output(text, expected_output):
    kant_rag = KantRAG()
    output = kant_rag._process_output(text=text)
    assert output == expected_output


@pytest.mark.parametrize(
    "input_response, rewritten_response",
    [
        ("This is the original response.", "This is the rewritten response."),
        ("Another original response.", "Another rewritten response."),
    ],
)
@patch("kant_rag.modeling.rag_model.ConstitutionalChain.from_llm")
@patch("kant_rag.modeling.rag_model.ConstitutionalChain.get_principles")
def test_rewrite_response(
    mock_get_principles, mock_from_llm, input_response, rewritten_response
):
    mock_get_principles.return_value = MagicMock()

    mock_chain = MagicMock()
    mock_chain.run.return_value = rewritten_response
    mock_from_llm.return_value = mock_chain

    kant_rag = KantRAG(question="Dummy question")
    kant_rag.context = "Dummy context"

    kant_rag.prompt = PromptTemplate(
        input_variables=["context", "question"], template="{context}\n\n{question}"
    )
    kant_rag.llm = MagicMock(spec=OpenAI)

    result = kant_rag._rewrite_response()

    assert result == rewritten_response

    mock_get_principles.assert_called_once_with(PRINCIPLES)

    mock_chain.run.assert_called_once_with(
        question=kant_rag.question, context=kant_rag.context
    )


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
                "confidence": {
                    "average_log_probability": -0.2816,
                    "joint_probability_percentage": 0.0,
                    "perplexity": 1.3252,
                },
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
                "confidence": {
                    "average_log_probability": -0.2816,
                    "joint_probability_percentage": 0.0,
                    "perplexity": 1.3252,
                },
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
                "confidence": {
                    "average_log_probability": -0.2245,
                    "joint_probability_percentage": 0.5,
                    "perplexity": 1.1023,
                },
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
                "confidence": {
                    "average_log_probability": -0.2245,
                    "joint_probability_percentage": 0.5,
                    "perplexity": 1.1023,
                },
            },
        ),
    ],
)
@patch("kant_rag.modeling.rag_model.KantRAG._create_chain")
@patch.object(ChatOpenAI, "__call__")
@patch("kant_rag.modeling.rag_model.KantRAG._rewrite_response")
@patch("kant_rag.modeling.rag_model.EstimateConfidence.estimate")
def test_run(
    mock_estimate_confidence,
    mock_rewrite_response,
    mock_chat_openai_call,
    mock_create_chain,
    query,
    text,
    expected_output,
):
    mock_estimate_confidence.return_value = expected_output["confidence"]

    mock_rewrite_response.return_value = text["result"]

    mock_qa_chain = MagicMock()
    mock_qa_chain.return_value = text

    def side_effect_create_chain():
        setattr(kant_rag, "qa_chain", mock_qa_chain)
        setattr(kant_rag, "context", text["context"])
        setattr(kant_rag, "question", text["query"])
        setattr(kant_rag, "confidence_metrics", expected_output["confidence"])

    mock_create_chain.side_effect = side_effect_create_chain

    kant_rag = KantRAG(question=query)
    response = kant_rag.run()

    assert response == expected_output
