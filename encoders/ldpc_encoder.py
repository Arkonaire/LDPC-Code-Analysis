import numpy as np
import networkx as nx

from abc import abstractmethod


class LDPCEncoder:

    """Abstract class for LDPC Decoder."""
    def __init__(self, ldpc_graph: nx.Graph):

        """Initialization.
        Args:
            ldpc_graph: Tanner graph for the LDPC code.
        """
        self.var_nodes = [node for node, data in ldpc_graph.nodes(data='bipartite') if data == 0]
        self.check_nodes = [node for node, data in ldpc_graph.nodes(data='bipartite') if data == 1]
        self.num_checks = len(self.check_nodes)
        self.block_length = len(self.var_nodes)
        self.msg_length = self.block_length - self.num_checks

    @abstractmethod
    def encode(self, msg: np.ndarray) -> np.ndarray:

        """Encoder implementation.
        Args:
            msg: The message bits to be encoded.
        Returns:
            Encoded codeword.
        """
