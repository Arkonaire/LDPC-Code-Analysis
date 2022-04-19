import numpy as np
import networkx as nx

from copy import deepcopy
from .ldpc_decoder import LDPCDecoder


class MinSum(LDPCDecoder):

    """Gallager error correction scheme."""
    def __init__(self, ldpc_graph: nx.Graph, maxiter=30, self_correcting=False, scale=1, offset=0):

        """Initialization.
        Args:
            ldpc_graph: Tanner graph for the LDPC code.
            maxiter: Maximum no. of decoding iterations.
            self_correcting: Enable self correction.
            scale: Normalization factor in [0, 1]. Disabled if self correcting.
            offset: Offset for Min Sum step. Disabled if self correcting.
        """
        super().__init__(ldpc_graph)
        self.maxiter = maxiter
        self.self_correcting = self_correcting
        self.scale = scale if not self_correcting else 1
        self.offset = offset if not self_correcting else 0
        self.ldpc_graph = deepcopy(ldpc_graph)

    def decode(self, y: np.ndarray) -> np.ndarray:

        """Decoder implementation.
        Args:
            y: The received channel output.
        Returns:
            Decoded message.
        """

        # Initialize decoder input
        gamma = np.sign(y.reshape((-1, self.block_length)))
        for i in range(self.block_length):
            self.ldpc_graph.nodes[f'V{i}']['gamma'] = gamma[:, i]

        # Initialize variable to check messages
        for edge in self.ldpc_graph.edges:
            V, C = edge
            self.ldpc_graph[V][C]['alpha'] = self.ldpc_graph.nodes[V]['gamma']

        # Main loop
        for _ in range(self.maxiter):

            # Check to variable messages
            for edge in self.ldpc_graph.edges:
                V, C = edge
                in_messages = np.array([self.ldpc_graph[node][C]['alpha'] for node in self.ldpc_graph[C] if node != V])
                b1 = np.prod(np.sign(in_messages), axis=0)
                b2 = np.maximum(np.min(np.abs(in_messages), axis=0) - self.offset, 0)
                self.ldpc_graph[V][C]['beta'] = self.scale * b1 * b2

            # Variable to check messages
            for edge in self.ldpc_graph.edges:
                V, C = edge
                in_messages = np.array([self.ldpc_graph[V][node]['beta'] for node in self.ldpc_graph[V] if node != C])
                self.ldpc_graph[V][C]['alpha'] = np.sum(in_messages, axis=0)
                self.ldpc_graph[V][C]['alpha'] += self.ldpc_graph.nodes[V]['gamma']

            # A-posteriori information
            for V in self.var_nodes:
                in_messages = np.array([self.ldpc_graph[V][node]['beta'] for node in self.ldpc_graph[V]])
                self.ldpc_graph.nodes[V]['gamma_posterior'] = np.sum(in_messages, axis=0)
                self.ldpc_graph.nodes[V]['gamma_posterior'] += self.ldpc_graph.nodes[V]['gamma']
                self.ldpc_graph.nodes[V]['gamma_posterior'] = np.sign(self.ldpc_graph.nodes[V]['gamma_posterior'])
                self.ldpc_graph.nodes[V]['gamma_posterior'] = self.ldpc_graph.nodes[V]['gamma_posterior'].astype(int)

            # Evaluate codeword validity
            valid = True
            for C in self.check_nodes:
                in_messages = np.array([self.ldpc_graph.nodes[node]['gamma_posterior'] for node in self.ldpc_graph[C]])
                check = (np.prod(in_messages, axis=0) == 1)
                if not check.all():
                    valid = False
                    break

            # Check loop termination
            if valid:
                break

        # Evaluate final codeword
        x = np.array([self.ldpc_graph.nodes[f'V{i}']['gamma_posterior'] for i in range(self.block_length)])
        x = np.transpose(x)
        x = ((1 - x) / 2).astype(int)
        return x
