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
