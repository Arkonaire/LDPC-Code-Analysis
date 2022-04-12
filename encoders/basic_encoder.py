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
        self.par_matrix = self._build_par_check()
        self.gen_matrix = self._build_generator()
        self.msg_length = self.gen_matrix.shape[0]

    def _build_par_check(self) -> np.ndarray:

        """Build the parity check matrix from the LDPC graph.
        Returns:
            The parity check matrix
        """

        # Build parity check matrix
        parity_check = np.zeros((self.num_checks, self.block_length), dtype=int)
        for j in range(self.block_length):
            for i in [int(x[1:]) for x in self.ldpc_graph[f'V{j}']]:
                parity_check[i, j] = 1

        # Return output
        return parity_check

    def _build_generator(self) -> np.ndarray:

        """Build the generator matrix from the LDPC graph.
        Returns:
            The generator matrix
        """

        # Convert parity check matrix to row echelon form
        parity_check = np.array(self.par_matrix)
        permutation = np.arange(self.block_length)
        for i in range(self.num_checks):

            # Identify pivot column
            pivot_cols = np.where(parity_check[i, :] == 1)[0] if i < parity_check.shape[0] else None
            while pivot_cols is not None and len(pivot_cols) == 0:
                parity_check = np.delete(parity_check, i, axis=0)
                pivot_cols = np.where(parity_check[i, :] == 1)[0] if i < parity_check.shape[0] else None

            # Termination check
            if pivot_cols is None:
                break

            # Apply column swap
            j = pivot_cols[0]
            parity_check[:, [i, j]] = parity_check[:, [j, i]]
            permutation[[i, j]] = permutation[[j, i]]

            # Apply row operations
            for j in range(parity_check.shape[0]):
                if j != i and parity_check[j, i] == 1:
                    parity_check[j, :] = ((parity_check[j, :] + parity_check[i, :]) % 2)

        # Build generator matrix
        P = parity_check[:, parity_check.shape[0]:]
        G = np.concatenate((np.transpose(P), np.eye(P.shape[1], dtype=int)), axis=1)
        gen_matrix = np.zeros_like(G)
        gen_matrix[:, permutation] = G
        return gen_matrix

    def encode(self, msg: np.ndarray) -> np.ndarray:

        """Encoder implementation.
        Args:
            msg: The message bits to be encoded.
        Returns:
            Encoded codeword.
        """
        msg = msg.reshape((-1, self.msg_length))
        x = (np.dot(msg, self.gen_matrix) % 2).astype(int).reshape((-1, self.block_length))
        return x
