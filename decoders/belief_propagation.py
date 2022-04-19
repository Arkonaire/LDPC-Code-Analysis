import numpy as np
import networkx as nx

from copy import deepcopy
from channels import ClassicalChannel
from .ldpc_decoder import LDPCDecoder


class BeliefPropagation(LDPCDecoder):

    """Belief Propagation error correction scheme."""
    def __init__(self, ldpc_graph: nx.Graph, channel: ClassicalChannel, maxiter=30):

        """Initialization.
        Args:
            ldpc_graph: Tanner graph for the LDPC code.
            channel: Communication channel for LLR evaluation.
            maxiter: Maximum no. of decoding iterations.
        """
        super().__init__(ldpc_graph)
        self.channel = channel
        self.maxiter = maxiter
        self.ldpc_graph = deepcopy(ldpc_graph)
        np.seterr(divide='ignore', invalid='ignore')

    def decode(self, y: np.ndarray) -> np.ndarray:

        """Decoder implementation.
        Args:
            y: The received channel output.
        Returns:
            Decoded message.
        """

        # Initialize decoder input
        phi = lambda z: -np.log(np.tanh(z/2))
        gamma = np.sign(y.reshape((-1, self.block_length)))
        gamma = self.channel.log_likelihood_ratio(gamma)
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
                b2 = phi(np.sum(phi(np.abs(in_messages)), axis=0))
                self.ldpc_graph[V][C]['beta'] = b1 * b2

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
