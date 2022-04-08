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
