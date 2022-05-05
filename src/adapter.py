from typing import Any
import networkx as nx

def mapper_to_networkx(graph) -> Any:
    file_name = 'g.net'
    G = nx.Graph(name = file_name)

    for i, vertice in enumerate(graph.vs):
        G.add_node(i)

    for edge in graph.es:
        edge1 = edge.source
        edge2 = edge.target

        G.add_edge(edge1, edge2)

    return G