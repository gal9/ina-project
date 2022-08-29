import numpy as np
import pandas as pd
import os
import time
import itertools
import open3d as o3d

from gtda.plotting import plot_point_cloud
from gtda.plotting import plot_diagram
import matplotlib.pyplot as plt
import seaborn as sn
import dataframe_image as dfi
from matplotlib.patches import Patch

from gtda.homology import VietorisRipsPersistence
from gtda.mapper import (
    CubicalCover,
    OneDimensionalCover,
    make_mapper_pipeline,
    Projection,
    plot_static_mapper_graph,
    plot_interactive_mapper_graph,
    MapperInteractivePlotter
)

from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn import svm

from src.feature_vectors import create_feature_vector
from src import adapter

np.random.seed(13)

def get_ply_files(folder):
    files = list(filter(lambda file: file.split('.')[-1]=='ply', os.listdir(folder)))
    files = list(map(lambda file: os.path.join(folder, file),files))
    return files

def label_groups(files, k):
    group_sizes = [len([f for f in os.listdir('data/'+file) if f[-3:]=='ply']) for file in files]
    group_sizes = [sum(group_sizes[:k]), group_sizes[k], sum(group_sizes[k+1:])]
    labels = np.zeros(sum(group_sizes))
    labels[group_sizes[0]:group_sizes[0]+group_sizes[1]] = 1
    return labels

def get_data(k):
    ply_files  = get_ply_files('data/tablesPly')
    ply_files += get_ply_files('data/chairsPly')
    ply_files += get_ply_files('data/octopusPly')
    ply_files += get_ply_files('data/spidersPly')

    files = ['tablesPly','chairsPly', 'octopusPly', 'spidersPly']
    labels = label_groups(files, k)
    majority_classifier = (1 - sum(labels)/len(labels))

    pcd = [o3d.io.read_point_cloud(file) for file in ply_files]
    pcd = [np.asarray(pc.points) for pc in pcd]
    
    shuffle_index = np.random.permutation(np.arange(0, len(labels)))
    labels = np.array(labels)
    pcd = np.array(pcd, dtype=object)
    labels = labels[shuffle_index]
    pcd = pcd[shuffle_index]
    
    return pcd, labels, majority_classifier

def mapper_pipeline(filter_func='Projection', cover='CubicalCover4-0.1', clusterer='DBSCAN10'):
    homology_dimensions = [0, 1, 2]

    persistence = VietorisRipsPersistence(
        metric="euclidean",
        homology_dimensions=homology_dimensions,
        n_jobs=6,
        collapse_edges=True,
    )
    
    if filter_func == 'PCA':
        filter_func = PCA(n_components=2)
    elif filter_func == 'Projection':
        filter_func = Projection(columns=[0,1,2])
    else:
        raise NameError('Name different filter function.')
    
    if cover[:12] == 'CubicalCover':
        n, f = cover[12:].split("-")
        cover = CubicalCover(n_intervals=int(n), overlap_frac=float(f))
    elif cover[:19] == 'OneDimensionalCover':
        n_intervals, overlap_frac = cover[19:].split("-")
        cover = OneDimensionalCover(kind='uniform', n_intervals=int(n), overlap_frac=float(f))
    else:
        raise NameError('Name different filter cover.')

    if clusterer[:6] == 'DBSCAN':
        clusterer = DBSCAN(eps=int(clusterer[6:]), metric="chebyshev")
    else:
        raise NameError('Name different filter clusterer.')

    n_jobs = 1

    pipe = make_mapper_pipeline(
        filter_func=filter_func,
        cover=cover,
        clusterer=clusterer,
        verbose=False,
        n_jobs=n_jobs,
    )
    
    return pipe

def persistence_homology():
    homology_dimensions = [0, 1, 2]
    persistence = VietorisRipsPersistence(
        metric="euclidean",
        homology_dimensions=homology_dimensions,
        n_jobs=6,
        collapse_edges=True,
    )
    return persistence

def feature_vectors(pcd, pipe):
    persistence = persistence_homology()
    
    entropy_feature_vectors = []
    feature_vectors = []
    start = time.time()
    for i, pc in enumerate(pcd):
        print('\r', f"{int((i/len(pcd))*100)}%", end="")
        e_fv, fv = create_feature_vector(pc, pipe, persistence)
        entropy_feature_vectors.append(e_fv)
        feature_vectors.append(fv)
    end = time.time()
    print()
    
    return feature_vectors, entropy_feature_vectors

def accuracy_scores(feature_vectors, entropy_feature_vectors, labels):
    num_features = len(feature_vectors[0])
    best_scores = []

    for homology_idx in range(3):
        final_fvs = []

        for entropy_fv in entropy_feature_vectors:
            final_fvs.append(entropy_fv[homology_idx])

        clf = svm.SVC(kernel='linear', C=1, random_state=42)
        scores = cross_val_score(clf, final_fvs, labels, cv=10, scoring="roc_auc")
        y_pred = cross_val_predict(clf, final_fvs, labels, cv=10)
        conf_mat = confusion_matrix(labels, y_pred)
        best_scores.append((scores.mean(), "h"+str(homology_idx+1), conf_mat))
        
        for number_of_additional_features in range(1,4):
            combinations = list(itertools.combinations(range(num_features), number_of_additional_features))

            for combination in combinations:
                final_fvs = []
                for fv_idx, entropy_fv in enumerate(entropy_feature_vectors):
                    extracted_fv = [x for x in entropy_fv[homology_idx]]
                    extracted_fv += [feature_vectors[fv_idx][i] for i in combination]
                    final_fvs.append(extracted_fv)

                scores = cross_val_score(clf, final_fvs, labels, cv=10, scoring="roc_auc")
                y_pred = cross_val_predict(clf, final_fvs, labels, cv=10)
                conf_mat = confusion_matrix(labels, y_pred)
                best_scores.append((scores.mean(), ''.join([str(x)+" " for x in combination]) + "h" + str(homology_idx+1), conf_mat))

    best_scores.sort(reverse=True)
    
    best_scores_2 = []
    for number_of_additional_features in range(1,6):
        combinations = list(itertools.combinations(range(num_features), number_of_additional_features))

        for combination in combinations:
            final_fvs = []
            for fv_idx, fv in enumerate(feature_vectors):
                extracted_fv = [fv[i] for i in combination]
                final_fvs.append(extracted_fv)

            scores = cross_val_score(clf, final_fvs, labels, cv=10, scoring="roc_auc")
            y_pred = cross_val_predict(clf, final_fvs, labels, cv=10)
            conf_mat = confusion_matrix(labels, y_pred)
            best_scores_2.append((scores.mean(), ''.join([str(x)+" " for x in combination]), conf_mat))


    best_scores_2.sort(reverse=True)
    return best_scores, best_scores_2
    

def model_accuracy(k, filter_func='Projection', cover='CubicalCover4-0.1', clusterer='DBSCAN10'):
    print("- processing data")
    pcd, labels, majority_classifier = get_data(k)
    pipe = mapper_pipeline(filter_func, cover, clusterer)
    print("- computing feature vectors")
    feature_vect, entropy = feature_vectors(pcd, pipe)
    print("- calculating accuracy scores")
    scores, scores_wo_homology = accuracy_scores(feature_vect, entropy, labels)
    return scores, scores_wo_homology, majority_classifier

def visualize_scores(best_scores, majority_classifier):
    data = [(score, comb, mapper) for (score, comb, cm), mapper in best_scores if score >= majority_classifier]
    mapper_settings, col = set([x[2][1:] for x in data]), [(0., 0., 1), (0.3, 0., 0.7), (0.4, 0., 0.6), (0.5, 0., 0.5)]
    colors = {x: col[i] for i, x in enumerate(mapper_settings)}
    legend_elements = [Patch(facecolor=c, label=s) for s, c in colors.items()]
    
    fig = plt.figure(figsize = (12, 5))
    for x in data:
        plt.bar(x[1], x[0], color = colors[x[2][1:]], width = 0.4)
    plt.bar("baseline", majority_classifier, color ='red', width = 0.4)
    plt.ylim(0.5, 1)
    plt.xticks(rotation = 45)
    plt.xlabel("Feature vectors")
    plt.ylabel("Accuracy")
    plt.title("Method accuracy using different combinations of parameters")
    plt.legend(handles=legend_elements, loc='upper right', title="Mapper settings")
    plt.savefig('results/barplots/best_results.png')
    plt.show()


def show_confusion_matrix(cm):
    df_cm = pd.DataFrame(cm, index = ['True','False'], columns = ['True','False'])
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size
    plt.show()