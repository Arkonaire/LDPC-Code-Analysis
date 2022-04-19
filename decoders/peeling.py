import numpy as np
import networkx as nx

from copy import deepcopy
from .ldpc_decoder import LDPCDecoder


class Peeling(LDPCDecoder):

    """Peeling error correction scheme."""
    def __init__(self, ldpc_graph: nx.Graph):

        """Initialization.
        Args:
            ldpc_graph: Tanner graph for the LDPC code.
        """
        super().__init__(ldpc_graph)
        self.ldpc_graph = deepcopy(ldpc_graph)

    def _decode_internal(self, y: np.ndarray) -> np.ndarray:

        """Internal decoder implementation for each trial.
        Args:
            y: The received channel output.
        Returns:
            Decoded message.
        """

        # Initialization
        ldpc_graph = deepcopy(self.ldpc_graph)
        omega = np.ones(self.num_checks)
        x = np.array(y)
        for i, val in enumerate(y):
            if val != 0:
                check_indices = [int(node[1:]) for node in ldpc_graph[f'V{i}']]
                omega[check_indices] = omega[check_indices] * val
                ldpc_graph.remove_node(f'V{i}')

        # Main Loop
        check_node = next((node for node in self.check_nodes if len(ldpc_graph[node]) == 1), None)
        while check_node is not None:
            var_node = next(iter(ldpc_graph[check_node]))
            var_index = int(var_node[1:])
            chk_index = int(check_node[1:])
            x[var_index] = omega[chk_index]
            check_indices = [int(node[1:]) for node in ldpc_graph[var_node]]
            omega[check_indices] = omega[check_indices] * x[var_index]
            ldpc_graph.remove_node(var_node)
            check_node = next((node for node in self.check_nodes if len(ldpc_graph[node]) == 1), None)

        # Evaluate final codeword
        x = ((1 - x) / 2).astype(int)
        return x

    def decode(self, y: np.ndarray) -> np.ndarray:

        """Decoder implementation.
        Args:
            y: The received channel output.
        Returns:
            Decoded message.
        """

        # Decode all inputs
        y = np.sign(y.reshape((-1, self.block_length))).astype(int)
        x = np.zeros_like(y)
        for i in range(y.shape[0]):
            x[i, :] = self._decode_internal(y[i, :])

        # Return outputs
        return x
