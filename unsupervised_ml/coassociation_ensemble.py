import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import normalized_mutual_info_score
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram, set_link_color_palette
from IPython.display import display, HTML


def create_coassociation_matrix(labels_dict, n_samples):
    """
    Create a co-association matrix from a dictionary of clustering label arrays.

    Parameters:
        labels_dict (dict): clustering method -> array of cluster labels
        n_samples (int): number of samples

    Returns:
        np.ndarray: co-association matrix of shape (n_samples, n_samples)
    """
    matrix = np.zeros((n_samples, n_samples))
    n_clusterings = len(labels_dict)
    for labels in labels_dict.values():
        for i in range(n_samples):
            matrix[i, i] += 1
            for j in range(i + 1, n_samples):
                if labels[i] == labels[j]:
                    matrix[i, j] += 1
                    matrix[j, i] += 1
    return matrix / n_clusterings


def perform_coassociation_ensemble(labels_dict, data_df, threshold=0.7):
    n_samples = len(data_df)
    coassoc = create_coassociation_matrix(labels_dict, n_samples)
    distance_matrix = 1 - coassoc
    np.fill_diagonal(distance_matrix, 0)
    condensed_dist = squareform(distance_matrix)
    linkage_matrix = linkage(condensed_dist, method='ward')
    n_clusters = len(np.where(linkage_matrix[:, 2] > threshold * np.max(linkage_matrix[:, 2]))[0]) + 1
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = clustering.fit_predict(distance_matrix)
    return labels, n_clusters, coassoc


def plot_coassociation_heatmap(coassoc_matrix, labels):
    sample_n = min(10, len(labels))
    indices = random.sample(range(len(labels)), sample_n)
    names = [labels[i] for i in indices]
    sampled_matrix = coassoc_matrix[np.ix_(indices, indices)]
    df = pd.DataFrame(sampled_matrix, index=names, columns=names)
    plt.figure(figsize=(20, 16))
    sns.heatmap(df, cmap="YlOrRd", annot=True, fmt='.3f', cbar_kws={'label': 'Co-association frequency'}, linewidths=0.5)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.title("Co-association Matrix Heatmap")
    plt.tight_layout()
    plt.show()


def plot_ensemble_dendrogram(data, cluster_labels, company_labels, title='Ensemble Clustering Dendrogram'):
    Z = linkage(data, method='ward')
    n_clusters = len(np.unique(cluster_labels))
    color_threshold = Z[-(n_clusters - 1), 2]
    set_link_color_palette([f'C{i}' for i in range(n_clusters)])
    n_leaves = Z.shape[0] + 1
    fig_width = max(20, n_leaves * 0.3)
    plt.figure(figsize=(fig_width, 10))
    dendrogram(Z, labels=company_labels[:n_leaves], leaf_rotation=90, leaf_font_size=8, color_threshold=color_threshold)
    plt.title(title)
    plt.xlabel('Company')
    plt.ylabel('Distance')
    plt.xticks(rotation=90, ha='center', fontsize=8)
    plt.tight_layout()
    display(HTML(f"""
    <div style=\"width: 100%; overflow-x: scroll;\">
        <img src=\"ensemble_dendrogram_majority_voting.png\" style=\"max-width: none; width: {fig_width*100}px;\">
    </div>
    """))
