from dataclasses import dataclass

import numpy as np


@dataclass
class EstimateConfidence:
    """
    Class for estimating confidence using output token log probabilities

    :param logprobs: list of log probabilities for individual tokens
    """

    logprobs: list[float]

    def _calculate_average(self) -> float:
        """
        Calculates average of log probabilities of the output tokens

        :returns average log probabilities of the output tokens
        """

        return round(sum(self.logprobs) / len(self.logprobs), 4)

    def _calculate_joint_probability_percentage(self) -> float:
        """
        Calculates joint probability of all tokens as a percentage

        :returns joint probability of all tokens as a percentage
        """
        return round(np.exp(sum(self.logprobs)) * 100, 4)

    def _calculate_perplexity(self) -> float:
        """
        Calculates perplexity of output tokens

        :returns perplexity of output tokens
        """
        # Convert log probabilities to probabilities
        probs = np.exp(self.logprobs)

        # Calculate sentence probability
        sentence_probability = probs.prod()

        # Normalize sentence probability by the number of words to handle varying lengths
        normalized_sentence_probs = sentence_probability ** (1 / len(probs))

        return round(1 / normalized_sentence_probs, 4)

    def estimate(self) -> dict[float, float, float]:
        """
        Calculates confidence metrics

        :returns dictionary of metrics
        """
        return {
            "average_log_probability": round(self._calculate_average(), 4),
            "joint_probability_percentage": round(
                self._calculate_joint_probability_percentage(), 4
            ),
            "perplexity": round(self._calculate_perplexity(), 4),
        }
