import networkx as nx


def read_ldpc_graph(filepath: str) -> nx.Graph:

    """Read LPDC data from file.
    Args:
        filepath: Path to file.
    Returns:
        Tanner Graph for LDPC code.
    """

    # Read data from file
    with open(filepath) as file:
        data = file.read().rstrip('\n \t').split('\n')
        data = [line.rstrip(' \t') for line in data]
        data = [[int(x) for x in line.split(' ')] for line in data]

    # Build LDPC graph
    ldpc_graph = nx.Graph()
    ldpc_graph.add_nodes_from([(f'V{i}', {'bipartite': 0}) for i in range(data[0][0] + data[0][1])])
    ldpc_graph.add_nodes_from([(f'C{i}', {'bipartite': 1}) for i in range(data[0][0])])
    for j in range(len(data) - 1):
        ldpc_graph.add_edges_from([(f'V{j}', f'C{i - 1}') for i in data[j + 1][1:]])

    # Return LDPC graph
    return ldpc_graph
