import numpy as np
import networkx as nx

from copy import deepcopy
from decoders.ldpc_decoder import LDPCDecoder


class GallagerB(LDPCDecoder):

    """Gallager-B error correction scheme."""
    def __init__(self, ldpc_graph: nx.Graph, maxiter=50, vote_threshold=1):

        """Initialization.
        Args:
            ldpc_graph: Tanner graph for the LDPC code.
            maxiter: Maximum no. of decoding iterations.
            vote_threshold: Voting Threshold for variable nodes.
        """
        self.maxiter = maxiter
        self.vote_threshold = vote_threshold
        self.ldpc_graph = deepcopy(ldpc_graph)
        super().__init__(self.ldpc_graph)

    def decode(self, y: np.ndarray) -> np.ndarray:

        """Decoder implementation.
        Args:
            y: The received channel output.
        Returns:
            Decoded message.
        """

        # Initialize decoder input
        gamma = 1 - 2 * y.reshape((-1, self.block_length))
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
                in_messages = np.array([self.ldpc_graph[node][C]['alpha'] for node in self.ldpc_graph[C]])
                self.ldpc_graph[V][C]['beta'] = np.prod(in_messages, axis=0) * self.ldpc_graph[V][C]['alpha']

            # Variable to check messages
            for edge in self.ldpc_graph.edges:
                V, C = edge
                in_messages = np.array([self.ldpc_graph[V][node]['beta'] for node in self.ldpc_graph[V]])
                self.ldpc_graph[V][C]['vote'] = np.sum(in_messages, axis=0) - self.ldpc_graph[V][C]['beta']
                self.ldpc_graph[V][C]['vote'] += self.ldpc_graph.nodes[V]['gamma']
                vote_check = (np.abs(self.ldpc_graph[V][C]['vote']) >= self.vote_threshold).astype(int)
                self.ldpc_graph[V][C]['alpha'] = (1 - vote_check) * self.ldpc_graph.nodes[V]['gamma']
                self.ldpc_graph[V][C]['alpha'] += vote_check * np.sign(self.ldpc_graph[V][C]['vote'])

            # A-posteriori information
            for V in self.var_nodes:
                in_messages = np.array([self.ldpc_graph[V][node]['beta'] for node in self.ldpc_graph[V]])
                self.ldpc_graph.nodes[V]['vote'] = self.ldpc_graph.nodes[V]['gamma'] + np.sum(in_messages, axis=0)
                vote_check = (self.ldpc_graph.nodes[V]['vote'] != 0).astype(int)
                self.ldpc_graph.nodes[V]['gamma_posterior'] = (1 - vote_check) * self.ldpc_graph.nodes[V]['gamma']
                self.ldpc_graph.nodes[V]['gamma_posterior'] += vote_check * np.sign(self.ldpc_graph.nodes[V]['vote'])

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
