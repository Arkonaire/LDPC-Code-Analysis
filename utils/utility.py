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
    ldpc_graph.add_nodes_from([(f'V{i}', {'bipartite': 0}) for i in range(data[0][0])])
    ldpc_graph.add_nodes_from([(f'C{i}', {'bipartite': 1}) for i in range(data[0][1])])
    for j in range(data[0][0]):
        ldpc_graph.add_edges_from([(f'V{j}', f'C{i - 1}') for i in data[j + 1][1:]])

    # Verify LDPC graph
    for j in range(data[0][1]):
        edgelist_a = set([f'V{i - 1}' for i in data[j + data[0][0] + 1][1:]])
        edgelist_b = set(ldpc_graph[f'C{j}'])
        assert edgelist_b == edgelist_a, f'Invalid Tanner Graph. Check node C{j} is inconsistent.'

    # Return LDPC graph
    return ldpc_graph
