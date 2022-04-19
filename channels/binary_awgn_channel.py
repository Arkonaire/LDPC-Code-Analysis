import numpy as np
from .channel_base import ClassicalChannel


class BinaryAWGNChannel(ClassicalChannel):

    """Python class for modelling a Binary AWGN Channel."""
    def __init__(self, sigma=0.5):

        """Initialize channel
        Args:
            sigma: Standard deviation of AWGN noise.
        """
        self.sigma = sigma

    def transmit(self, x: np.ndarray) -> np.ndarray:

        """Transmit data.
        Args:
            x: Codeword to be transmitted.
        Returns:
            Channel output.
        """
        e = np.random.normal(0, self.sigma, size=x.shape)
        y = 1 - 2*x + e
        return y

    def log_likelihood_ratio(self, y: np.ndarray) -> np.ndarray:

        """Evaluate the Log Likelihood Ratio for the inputs given the channel output.
        Args:
            y: Received codeword.
        Returns:
            LLR of the inputs given received y.
        """
        return 2 * y / (self.sigma ** 2)
