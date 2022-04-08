import numpy as np
import networkx as nx

from copy import deepcopy
from encoders import LDPCEncoder


class BasicEncoder(LDPCEncoder):

    """Basic LDPC encoding via the generator matrix."""
    def __init__(self, ldpc_graph: nx.Graph):

        """Initialization.
        Args:
            ldpc_graph: Tanner graph for the LDPC code.
        """
        super().__init__(ldpc_graph)
        self.ldpc_graph = deepcopy(ldpc_graph)
        self.gen_matrix = self.build_generator()

    def build_generator(self) -> np.ndarray:

        """Build the generator matrix from the LDPC graph.
        Returns:
            The generator matrix
        """
        # TODO

    def encode(self, msg: np.ndarray) -> np.ndarray:

        """Encoder implementation.
        Args:
            msg: The message bits to be encoded.
        Returns:
            Encoded codeword.
        """
        return (np.dot(msg, self.gen_matrix) % 2).astype(int)
