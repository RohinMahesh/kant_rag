import pytest
from kant_rag.evaluation.evaluate import EvaluateRAG


@pytest.mark.parametrize(
    "input_data, expected_exception",
    [
        (
            {
                "question": ["What is AI?"],
                "answer": ["Artificial Intelligence"],
                "contexts": [["AI is a branch of computer science."]],
                "ground_truth": ["AI is a branch of computer science."],
            },
            None,
        ),
        (
            {
                "question": ["What is AI?"],
                "answer": ["Artificial Intelligence"],
                "contexts": [["AI is a branch of computer science."]],
            },
            AssertionError,
        ),
        (
            {
                "answer": ["Artificial Intelligence"],
                "contexts": [["AI is a branch of computer science."]],
                "ground_truth": ["AI is a branch of computer science."],
            },
            AssertionError,
        ),
    ],
)
def test_evaluate_rag(input_data, expected_exception):
    if expected_exception:
        with pytest.raises(expected_exception):
            EvaluateRAG(data=input_data)
    else:
        scores = EvaluateRAG(data=input_data)
