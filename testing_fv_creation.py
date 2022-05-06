from src.feature_vectors import create_feature_vector
import time
start = time.time()


# Data wrangling
import numpy as np
import pandas as pd  # Not a requirement of giotto-tda, but is compatible with the gtda.mapper module
import os
import open3d as o3d

# Data viz
from gtda.plotting import plot_point_cloud
from gtda.plotting import plot_diagram

# TDA magic
from gtda.homology import VietorisRipsPersistence
from gtda.mapper import (
    CubicalCover,
    make_mapper_pipeline,
    Projection,
    plot_static_mapper_graph,
    plot_interactive_mapper_graph,
    MapperInteractivePlotter
)

# ML tools
from sklearn import datasets
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

def get_ply_files(folder):
    files = list(filter(lambda file: file.split('.')[-1]=='ply', os.listdir(folder)))
    files = list(map(lambda file: os.path.join(folder, file),files))
    return files

ply_files  = get_ply_files('data/tablesPly')
ply_files += get_ply_files('data/chairsPly')
ply_files += get_ply_files('data/octopusPly')
ply_files += get_ply_files('data/spidersPly')

labels, index = np.zeros(len(ply_files)), len(os.listdir('data/tablesPly')), 
index2 = index + len(os.listdir('data/chairsPly'))
labels[index:index2] = 1
index, index2 = index2, index2 + len(os.listdir('data/octopusPly'))
labels[index:index2] = 2
labels[index2:] = 3

filter_func = PCA(n_components=2)

cover = CubicalCover(n_intervals=4, overlap_frac=0.08)
#cover = OneDimensionalCover(kind='uniform', n_intervals=10, overlap_frac=0.1)

clusterer = DBSCAN(eps=10, metric="chebyshev")

n_jobs = 1

pipe = make_mapper_pipeline(
    filter_func=filter_func,
    cover=cover,
    clusterer=clusterer,
    verbose=False,
    n_jobs=n_jobs,
)

homology_dimensions = [0, 1, 2]

# Collapse edges to speed up H2 persistence calculation!
persistence = VietorisRipsPersistence(
    metric="euclidean",
    homology_dimensions=homology_dimensions,
    n_jobs=6,
    collapse_edges=True,
)

pcd = [o3d.io.read_point_cloud(file) for file in ply_files]
pcd = [np.asarray(pc.points) for pc in pcd]


entropy_fv, fv = create_feature_vector(pcd[1], pipe, persistence)

print(entropy_fv)

print(fv)