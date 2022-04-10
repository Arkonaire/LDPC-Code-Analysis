import numpy as np
import networkx as nx

from copy import deepcopy
from encoders.ldpc_encoder import LDPCEncoder


class BasicEncoder(LDPCEncoder):

    """Basic LDPC encoding via the generator matrix."""
    def __init__(self, ldpc_graph: nx.Graph):

        """Initialization.
        Args:
            ldpc_graph: Tanner graph for the LDPC code.
        """
        super().__init__(ldpc_graph)
        self.ldpc_graph = deepcopy(ldpc_graph)
        self.gen_matrix, self.permutation = self.build_generator()
        self.inv_permutation = np.zeros_like(self.permutation)
        self.inv_permutation[self.permutation] = np.arange(self.block_length)

    def build_generator(self) -> (np.ndarray, np.ndarray):

        """Build the generator matrix from the LDPC graph.
        Returns:
            The generator matrix
        """

        # Build parity check matrix
        parity_check = np.zeros((self.num_checks, self.block_length), dtype=int)
        for j in range(self.block_length):
            for i in [int(x[1:]) for x in self.ldpc_graph[f'V{j}']]:
                parity_check[i, j] = 1

        # Convert parity check matrix to row echelon form
        permutation = np.arange(self.block_length)
        for i in range(self.num_checks):
            j = np.where(parity_check[i, :] == 1)[0][0]
            parity_check[:, [i, j]] = parity_check[:, [j, i]]
            permutation[[i, j]] = permutation[[j, i]]
            for j in range(self.num_checks):
                if j != i and parity_check[j, i] == 1:
                    parity_check[j, :] = ((parity_check[j, :] + parity_check[i, :]) % 2)

        # Build generator matrix
        P = parity_check[:, self.block_length:]
        gen_matrix = np.concatenate(np.transpose(P), np.eye(self.msg_length), axis=1)
        return gen_matrix.astype(int), permutation.astype(int)

    def encode(self, msg: np.ndarray) -> np.ndarray:

        """Encoder implementation.
        Args:
            msg: The message bits to be encoded.
        Returns:
            Encoded codeword.
        """
        msg = msg.reshape((-1, self.msg_length))
        x = (np.dot(msg, self.gen_matrix, keepdims=True) % 2).astype(int)
        x[:, self.inv_permutation] = x
        return x
