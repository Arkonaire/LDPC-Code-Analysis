import numpy as np
from .channel_base import ClassicalChannel


class BinaryErasureChannel(ClassicalChannel):

    """Python class for modelling a Binary Erasure Channel."""
    def __init__(self, epsilon=0):

        """Initialize channel
        Args:
            epsilon: Erasure probability.
        """
        self.epsilon = epsilon

    def transmit(self, x: np.ndarray) -> np.ndarray:

        """Transmit data.
        Args:
            x: Codeword to be transmitted.
        Returns:
            Channel output.
        """
        e = (np.random.rand(*x.shape) >= self.epsilon).astype(int)
        y = ((1 - 2 * x) * e).astype(int)
        return y

    def log_likelihood_ratio(self, y: np.ndarray) -> np.ndarray:

        """Evaluate the Log Likelihood Ratio for the inputs given the channel output.
        Args:
            y: Received codeword.
        Returns:
            LLR of the inputs given received y.
        """
        y = y.astype(int)
        llr = np.zeros_like(y, dtype=float)
        llr[y > 0] = np.inf
        llr[y < 0] = -np.inf
        return llr
