import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score, silhouette_samples
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import linkage, dendrogram, set_link_color_palette
from matplotlib import cm
import matplotlib.patches as mpl_patches


def evaluate_clustering_metrics(data, labels_dict):
    """Calculate Calinski-Harabasz and Davies-Bouldin scores for each clustering."""
    ch_scores = {}
    db_scores = {}
    for name, label in labels_dict.items():
        ch_scores[name] = calinski_harabasz_score(data, label)
        db_scores[name] = davies_bouldin_score(data, label)
    return pd.DataFrame({
        'Clustering Technique': list(ch_scores.keys()),
        'Calinski-Harabasz Index': list(ch_scores.values()),
        'Davies-Bouldin Index': list(db_scores.values())
    })


def list_clusters_with_labels(labels, label_names):
    clusters = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(label, []).append(label_names[idx])
    return {k: set(v) for k, v in clusters.items()}


def display_cluster_programs(cluster_program_lists):
    for name, clusters in cluster_program_lists.items():
        print(f"\n{name} Clusters:")
        for cluster, programs in clusters.items():
            print(f"  Cluster {cluster}:")
            for program in programs:
                print(f"    {program}")


def display_clusters_styled(cluster_program_lists):
    for name, clusters in cluster_program_lists.items():
        print(f"\n{name} Clusters:")
        cluster_table = pd.DataFrame.from_dict(clusters, orient='index').transpose()
        display(cluster_table.style.set_properties(**{'text-align': 'left'}))


def plot_cluster_heatmap(labels, data, method_name):
    labels = labels.astype(int)
    centers = np.array([data[labels == i].mean(axis=0) for i in np.unique(labels)])
    dist_matrix = cdist(centers, centers)
    plt.figure(figsize=(8, 6))
    sns.heatmap(dist_matrix, annot=True, cmap='viridis')
    plt.title(f'{method_name} Cluster Distance Heatmap')
    plt.show()


def plot_silhouette(data, labels, method_name):
    labels = labels.astype(int)
    silhouette_vals = silhouette_samples(data, labels)
    n_clusters = len(np.unique(labels))
    y_lower = 0
    yticks = []
    fig, ax = plt.subplots(figsize=(10, 7))
    for i in range(n_clusters):
        ith_cluster_vals = silhouette_vals[labels == i]
        ith_cluster_vals.sort()
        y_upper = y_lower + len(ith_cluster_vals)
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_vals, facecolor=color, edgecolor=color)
        yticks.append((y_lower + y_upper) / 2)
        y_lower = y_upper

    silhouette_avg = silhouette_score(data, labels)
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax.set_title(f"Silhouette Plot - {method_name}")
    ax.set_yticks(yticks)
    ax.set_yticklabels(range(n_clusters))
    ax.set_xlim([-0.1, 1])
    plt.show()


def plot_spider(data, labels, title):
    df = pd.DataFrame(data)
    num_vars = df.shape[1]
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    features = df.columns.tolist()
    cluster_means = df.groupby(labels).mean()
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    for idx, row in cluster_means.iterrows():
        values = row.tolist() + [row.tolist()[0]]
        ax.plot(angles, values, label=f'Cluster {idx}')
        ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features, fontsize=8)
    plt.legend(loc='upper right')
    plt.title(title)
    plt.show()


def plot_kmeans_tsne_centers(data, labels, hover_df):
    tsne = TSNE(n_components=2, random_state=42)
    tsne_data = tsne.fit_transform(data)
    tsne_df = pd.DataFrame(tsne_data, columns=['Component 1', 'Component 2'])
    tsne_df['Cluster'] = labels.astype(int)
    tsne_df = pd.concat([tsne_df, hover_df.reset_index(drop=True)], axis=1)
    centers = np.array([tsne_data[labels == i].mean(axis=0) for i in np.unique(labels)])
    fig = px.scatter(tsne_df, x='Component 1', y='Component 2', color='Cluster',
                     hover_data=['brand_program'],
                     title='K-Means Clustering with t-SNE')
    fig.add_trace(px.scatter(x=centers[:, 0], y=centers[:, 1], size=[10]*len(centers),
                             color_discrete_sequence=['red']).data[0])
    fig.update_layout(coloraxis_showscale=False)
    fig.show()


def plot_agglomerative_dendrogram(data, labels, n_clusters, company_labels):
    Z = linkage(data, method='ward')
    color_threshold = Z[-(n_clusters - 1), 2]
    set_link_color_palette([f'C{i}' for i in range(n_clusters)])
    fig, ax = plt.subplots(figsize=(60, 7))
    dendrogram(Z, labels=company_labels, leaf_rotation=90, leaf_font_size=10, ax=ax,
               color_threshold=color_threshold)
    plt.title('Agglomerative Clustering Dendrogram')
    plt.xticks(rotation=90, fontsize=8)
    plt.subplots_adjust(bottom=0.35, top=0.95, left=0.01, right=0.99)
    plt.show()


def plot_dbscan_clusters(data_2d, labels, core_samples_mask):
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    plt.figure(figsize=(10, 8))
    for k, col in zip(unique_labels, colors):
        class_mask = (labels == k)
        xy_core = data_2d[class_mask & core_samples_mask]
        xy_noncore = data_2d[class_mask & ~core_samples_mask]
        plt.plot(xy_core[:, 0], xy_core[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=14)
        plt.plot(xy_noncore[:, 0], xy_noncore[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6)
    plt.title('DBSCAN Core Samples and Noise')
    plt.show()


def plot_gmm_covariances(tsne_data, labels):
    labels = labels.astype(int)
    gmm = GaussianMixture(n_components=len(np.unique(labels)))
    gmm.means_ = np.array([tsne_data[labels == i].mean(axis=0) for i in np.unique(labels)])
    gmm.covariances_ = np.array([np.cov(tsne_data[labels == i].T) for i in np.unique(labels)])
    gmm.weights_ = np.array([np.mean(labels == i) for i in np.unique(labels)])
    w_factor = 0.2 / gmm.weights_.max()
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_data[:, 0], tsne_data[:, 1], c=labels, s=40, cmap='viridis')
    plt.colorbar(scatter, label='Cluster')
    for i, (pos, covar, w) in enumerate(zip(gmm.means_, gmm.covariances_, gmm.weights_)):
        if covar.ndim == 1:
            covar = np.diag(covar)
        U, s, _ = np.linalg.svd(covar)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
        ell = mpl_patches.Ellipse(pos, width, height, angle=angle, color=plt.cm.viridis(i / len(gmm.means_)), alpha=0.5)
        ell.set_edgecolor('black')
        ell.set_linewidth(1)
        plt.gca().add_artist(ell)
    plt.title('t-SNE + GMM Cluster Covariance Ellipses')
    plt.show()
