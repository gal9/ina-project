import networkx as nx
import numpy as np

from typing import List, Any, Tuple
from adapter import mapper_to_networkx 
from gtda.mapper import plot_static_mapper_graph
from gtda.diagrams import Amplitude, PersistenceEntropy

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
    entropy_feature_vector.append(PersistenceEntropy().fit_transform(diagram[None,:,:])[0].tolist())
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

    # ARTICULATION POINTS
    feature_vector.append(len(list(nx.articulation_points(networkx_graph))))

    # AVERAGE DEGREE
    feature_vector.append(2*m/n)

    # DENSITY
    feature_vector.append(2 * m / n / (n - 1))

    # NETWORK CENTRALITY
    feature_vector.append(nx.average_clustering(networkx_graph))

    # TODO add other metrices
    
    return entropy_feature_vector, feature_vector
