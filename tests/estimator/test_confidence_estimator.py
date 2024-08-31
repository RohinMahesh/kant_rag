import os
import sys

repo_path = os.path.abspath("/Users/rohinmahesh/Documents/GitHub/kant_rag")

if repo_path not in sys.path:
    sys.path.append(repo_path)

import pytest

from kant_rag.estimator.confidence_estimator import EstimateConfidence


@pytest.mark.parametrize(
    "logprobs, expected_output",
    [
        ([-0.2, -0.3, -0.5], -0.3333),
        ([-1.0, -1.0, -1.0], -1.0),
        ([-2.5, -1.5, -0.5], -1.5),
    ],
)
def test__calculate_average(logprobs, expected_output):
    estimation = EstimateConfidence(logprobs=logprobs)._calculate_average()
    assert estimation == expected_output


@pytest.mark.parametrize(
    "logprobs, expected_output",
    [
        ([-0.2, -0.3, -0.5], 36.7879),
        ([-1.0, -1.0, -1.0], 4.9787),
        ([-2.5, -1.5, -0.5], 1.1109),
    ],
)
def test__calculate_joint_probability_percentage(logprobs, expected_output):
    estimation = EstimateConfidence(
        logprobs=logprobs
    )._calculate_joint_probability_percentage()
    assert estimation == expected_output


@pytest.mark.parametrize(
    "logprobs, expected_output",
    [
        ([-0.2, -0.3, -0.5], 1.3956),
        ([-1.0, -1.0, -1.0], 2.7183),
        ([-2.5, -1.5, -0.5], 4.4817),
    ],
)
def test__calculate_perplexity(logprobs, expected_output):
    estimation = EstimateConfidence(logprobs=logprobs)._calculate_perplexity()
    assert estimation == expected_output


@pytest.mark.parametrize(
    "logprobs, expected_output",
    [
        (
            [-1.0, -1.0, -1.0],
            {
                "average_log_probability": -1.0,
                "joint_probability_percentage": 4.9787,
                "perplexity": 2.7183,
            },
        ),
        (
            [-2.5, -1.5, -0.5],
            {
                "average_log_probability": -1.5,
                "joint_probability_percentage": 1.1109,
                "perplexity": 4.4817,
            },
        ),
    ],
)
def test_estimate(logprobs, expected_output):
    estimation = EstimateConfidence(logprobs=logprobs).estimate()
    assert estimation == expected_output
