import networkx as nx
import networkx.algorithms.community as nx_comm
import numpy as np

from typing import List, Any, Tuple
from src.adapter import mapper_to_networkx
from gtda.mapper import plot_static_mapper_graph
from gtda.diagrams import Amplitude, PersistenceEntropy
from scipy import stats
import itertools


def get_degree_mixing(G):
    """Calculates degree mixing of graph"""
    x, y = [], []  # init values of k and k'
    # iterate over all edges
    for i, j in G.edges():
        # add degrees of i's and j's nodes to x and y: x<-[deg_i,deg_j], y<-[deg_j,deg_i]
        x.append(G.degree(i))
        y.append(G.degree(j))
        x.append(G.degree(j))
        y.append(G.degree(i))
    return stats.pearsonr(x, y)[0]


def find_cliques_size_k(G, k):
    all_cliques = set()
    for clique in nx.find_cliques(G):
        if len(clique) == k:
            all_cliques.add(tuple(sorted(clique)))
        elif len(clique) > k:
            for mini_clique in itertools.combinations(clique, k):
                all_cliques.add(tuple(sorted(mini_clique)))
    return len(all_cliques)


def get_number_of_leaves(G):
    count = 0
    for (node, val) in G.degree():
        if val == 1:
            count += 1
    return count


def get_max_closeness_centrality(G):
    centrality_nodes = list(nx.closeness_centrality(G).values())
    return max(centrality_nodes)


def get_max_eigenvector_centrality(G):
    centrality_nodes = list(nx.eigenvector_centrality(G).values())
    return max(centrality_nodes)


def get_max_betweenness_centrality(G):
    centrality_nodes = list(nx.betweenness_centrality(G).values())
    return max(centrality_nodes)


def get_90th_perc_closeness_centrality(G):
    ecentrality_nodes = list(nx.closeness_centrality(G).values())
    max_element = max(ecentrality_nodes)
    count_top_ecentrality = 0
    for i in ecentrality_nodes:
        if max_element * 0.9 <= i:
            count_top_ecentrality += 1
    return count_top_ecentrality


def get_90th_perc_betweenness_centrality(G):
    ecentrality_nodes = list(nx.betweenness_centrality(G).values())
    max_element = max(ecentrality_nodes)
    count_top_ecentrality = 0
    for i in ecentrality_nodes:
        if max_element * 0.9 <= i:
            count_top_ecentrality += 1
    return count_top_ecentrality


def get_avg_closeness_centrality(G):
    ecentrality_nodes = list(nx.closeness_centrality(G).values())
    return sum(ecentrality_nodes)/len(ecentrality_nodes)


def get_avg_betweenness_centrality(G):
    ecentrality_nodes = list(nx.betweenness_centrality(G).values())
    return sum(ecentrality_nodes)/len(ecentrality_nodes)


def create_feature_vector(point_cloud, pipe, persistence) -> Tuple[List[float], List[float]]:
    """
    A function that generates entropy feature vectors as well as network-based
    feature vectors and returns them seperately

    parameters:
        point_cloud: obvious,
        pipe: mapper pipeline,
        persistance: for homologies
    
    returns:
        entropy_feature_vector: a list containing 3 lists of
                                homologies (one list for each distance metrixx)
    """

    # ENTROPY FV
    entropy_feature_vector = []

    # Create a figure
    figure = plot_static_mapper_graph(pipe, point_cloud)

    # Compute entropy features
    mapped_points = np.array(list(x for x in zip(figure.data[1].x, figure.data[1].y) if None not in x))
    diagram = persistence.fit_transform(mapped_points[None, :, :])[0]
    entropy_feature_vector.append(PersistenceEntropy().fit_transform(diagram[None, :, :])[0].tolist())
    entropy_feature_vector.append(Amplitude(metric='wasserstein').fit_transform(diagram[None, :, :])[0].tolist())
    entropy_feature_vector.append(Amplitude(metric='bottleneck').fit_transform(diagram[None, :, :])[0].tolist())

    # ------------------- OTHER FEATURE VECTOR ----------------------------
    # The shape of the FV is [number_of_articulation_points, average degree, density, network centrality]

    # Create a graph to work on
    graph = pipe.fit_transform(point_cloud)
    networkx_graph = mapper_to_networkx(graph)

    n = networkx_graph.number_of_nodes()
    m = networkx_graph.number_of_edges()

    feature_vector = []

    # 0 NUMBER OF ARTICULATION POINTS
    feature_vector.append(len(list(nx.articulation_points(networkx_graph))))

    # 1 AVERAGE DEGREE
    feature_vector.append(2 * m / n)

    # 2 DENSITY
    feature_vector.append(2 * m / n / (n - 1))

    # 3 AVG. NETWORK CLUSTERING
    feature_vector.append(nx.average_clustering(networkx_graph))

    # 4 NUMBER OF NODES IN TOP 10% CLOSENESS CENTRALITY NORMALIZED
    feature_vector.append(float(get_90th_perc_closeness_centrality(networkx_graph)) / n)

    # 5 NUMBER OF NODES IN TOP 10% BETWEENNESS CENTRALITY NORMALIZED
    feature_vector.append(float(get_90th_perc_betweenness_centrality(networkx_graph)) / n)

    # 6 NUMBER OF CLIQUES OF 4 NORMALIZED
    feature_vector.append(float(find_cliques_size_k(networkx_graph, 4)) / n)

    # 7 ASSORTATIVITY COEFFICIENT
    feature_vector.append(nx.degree_assortativity_coefficient(networkx_graph))

    # 8 NUMBER OF LEAVES NORMALIZED
    feature_vector.append(float(get_number_of_leaves(networkx_graph)) / n)

    # 9 MAX CLOSENESS CENTRALITY
    feature_vector.append(get_max_closeness_centrality(networkx_graph))

    # 10 MAX BETWEENNESS CENTRALITY
    feature_vector.append(get_max_betweenness_centrality(networkx_graph))

    # 11 AVERAGE CLOSENESS CENTRALITY
    feature_vector.append(get_avg_closeness_centrality(networkx_graph))

    # 12 AVERAGE BETWEENNESS CENTRALITY
    feature_vector.append(get_avg_betweenness_centrality(networkx_graph))

    # 13 NUMBER OF LOUVAIN COMMUNITIES
    #feature_vector.append(len(nx_comm.louvain_communities(networkx_graph)))

    return entropy_feature_vector, feature_vector
