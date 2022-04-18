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
        return y
