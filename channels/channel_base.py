import numpy as np
from abc import abstractmethod


class ClassicalChannel:

    """Abstract class for classical channel."""

    @abstractmethod
    def transmit(self, x: np.ndarray) -> np.ndarray:

        """Transmit data.
        Args:
            x: Codeword to be transmitted.
        Returns:
            Channel output.
        """

    @abstractmethod
    def log_likelihood_ratio(self, y: np.ndarray) -> np.ndarray:

        """Evaluate the Log Likelihood Ratio for the inputs given the channel output.
        Args:
            y: Received codeword.
        Returns:
            LLR of the inputs given received y.
        """
