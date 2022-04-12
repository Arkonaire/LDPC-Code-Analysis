import numpy as np
import networkx as nx

from abc import abstractmethod


class LDPCDecoder:

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

    @abstractmethod
    def decode(self, y: np.ndarray) -> np.ndarray:

        """Decoder implementation.
        Args:
            y: The received channel output.
        Returns:
            Decoded message.
        """
