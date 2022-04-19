import numpy as np
from .channel_base import ClassicalChannel


class BinarySymmetricChannel(ClassicalChannel):

    """Python class for modelling a Binary Symmetric Channel."""
    def __init__(self, alpha=0):

        """Initialize channel
        Args:
            alpha: Bit flip probability.
        """
        self.alpha = alpha

    def transmit(self, x: np.ndarray) -> np.ndarray:

        """Transmit data.
        Args:
            x: Codeword to be transmitted.
        Returns:
            Channel output.
        """
        e = (np.random.rand(*x.shape) < self.alpha).astype(int)
        y = ((x + e) % 2).astype(int)
        return 1 - 2 * y

    def log_likelihood_ratio(self, y: np.ndarray) -> np.ndarray:

        """Evaluate the Log Likelihood Ratio for the inputs given the channel output.
        Args:
            y: Received codeword.
        Returns:
            LLR of the inputs given received y.
        """
        return y * np.log((1 - self.alpha) / self.alpha)
