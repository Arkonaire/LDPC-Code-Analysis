import numpy as np
from abc import abstractmethod


class ClassicalErrorCorrection:

    """Abstract class for classical error correction schemes."""
    def __init__(self, msg_length, block_length):

        """Initialization.
        Args:
            msg_length: Length of a message frame.
            block_length: Length of the codewords.
        """
        self.msg_length = msg_length
        self.block_length = block_length

    @abstractmethod
    def encode(self, msg: np.ndarray) -> np.ndarray:

        """Encoder implementation.
        Args:
            msg: The message bits to be encoded.
        Returns:
            Encoded codeword.
        """

    @abstractmethod
    def decode(self, y: np.ndarray) -> np.ndarray:

        """Encoder implementation.
        Args:
            y: The received channel output.
        Returns:
            Decoded message.
        """
