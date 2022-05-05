from typing import List, Any
from adapter import mapper_to_networkx

def create_feature_vector(point_cloud, pipe, persistance) -> List[float]:
    # FV of shape [Entropy, Entropy, Entropy]
    fetture_vector = []

    # Create a graph
    graph = pipe.fit_transform(pc)

    networkx_graph = mapper_to_networkx(graph)

    # Create a figure
    figure = plot_static_mapper_graph(pipe, pc)

    # Compute entropy features
    mapped_points = np.array(list(x for x in zip(fig.data[1].x, fig.data[1].y) if None not in x))
    diagram = persistence.fit_transform(mapped_points[None, :, :])[0]
    feature_vector += PersistenceEntropy().fit_transform(diagram[None,:,:])[0]

    return feature_vector
