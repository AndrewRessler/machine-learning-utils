import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, Birch, AffinityPropagation, MeanShift, estimate_bandwidth, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

def select_kmeans_k(data: np.ndarray, k_range=(2, 11), plot=True, random_state=42) -> int:
    """
    Determine the optimal number of clusters for KMeans using silhouette score.

    Args:
        data: 2D array-like data for clustering.
        k_range: Range of cluster counts to evaluate.
        plot: Whether to plot silhouette scores.
        random_state: Seed for reproducibility.

    Returns:
        Optimal number of clusters (int).
    """
    silhouette_scores = []
    for k in range(*k_range):
        kmeans = KMeans(n_clusters=k, random_state=random_state).fit(data)
        score = silhouette_score(data, kmeans.labels_)
        silhouette_scores.append(score)

    if plot:
        plt.plot(range(*k_range), silhouette_scores, marker='o')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.title('k-Means Silhouette Scores')
        plt.show()

    return np.argmax(silhouette_scores) + k_range[0]

def select_agglomerative_k(data: np.ndarray, k_range=(2, 11), plot=True) -> int:
    """
    Determine the optimal number of clusters for Agglomerative Clustering using silhouette score.

    Args:
        data: 2D array-like data for clustering.
        k_range: Range of cluster counts to evaluate.
        plot: Whether to plot silhouette scores.

    Returns:
        Optimal number of clusters (int).
    """
    silhouette_scores = []
    for k in range(*k_range):
        model = AgglomerativeClustering(n_clusters=k, metric='euclidean', linkage='ward')
        labels = model.fit_predict(data)
        silhouette_scores.append(silhouette_score(data, labels))

    if plot:
        plt.plot(range(*k_range), silhouette_scores, marker='o')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.title('Agglomerative Clustering Silhouette Scores')
        plt.show()

    return np.argmax(silhouette_scores) + k_range[0]

def estimate_dbscan_eps(data: np.ndarray, min_samples=5, plot=True) -> float:
    """
    Estimate the optimal epsilon value for DBSCAN using k-distance graph.

    Args:
        data: 2D array-like data for clustering.
        min_samples: Minimum samples for DBSCAN.
        plot: Whether to plot the k-distance graph.

    Returns:
        Optimal epsilon (float).
    """
    k = min_samples - 1
    neigh = NearestNeighbors(n_neighbors=k)
    nbrs = neigh.fit(data)
    distances, _ = nbrs.kneighbors(data)
    distances = np.sort(distances[:, k-1])

    first_derivative = np.diff(distances)
    second_derivative = np.diff(first_derivative)
    max_second_deriv_index = np.argmax(second_derivative) + 1
    optimal_eps = round(distances[max_second_deriv_index], 2)

    if plot:
        plt.plot(distances, label='k-distance')
        plt.axvline(x=max_second_deriv_index, color='r', linestyle='--', label='Optimal epsilon')
        plt.xlabel('Points sorted by distance')
        plt.ylabel(f'{k}-th Nearest Neighbor Distance')
        plt.title('k-Distance Graph')
        plt.legend()
        plt.show()

    return optimal_eps

def select_gmm_n_components(data: np.ndarray, n_range=(1, 11), plot=True, random_state=42) -> dict:
    """
    Determine optimal number of components for GMM using AIC and BIC.

    Args:
        data: 2D array-like data for clustering.
        n_range: Range of components to evaluate.
        plot: Whether to plot AIC/BIC.
        random_state: Seed for reproducibility.

    Returns:
        Dictionary with best AIC/BIC component count and full score lists.
    """
    bics, aics = [], []
    for n in range(*n_range):
        gmm = GaussianMixture(n_components=n, random_state=random_state).fit(data)
        bics.append(gmm.bic(data))
        aics.append(gmm.aic(data))

    if plot:
        plt.plot(range(*n_range), bics, label='BIC', marker='o')
        plt.plot(range(*n_range), aics, label='AIC', marker='o')
        plt.xlabel('Number of Components')
        plt.ylabel('BIC / AIC')
        plt.title('GMM AIC/BIC Scores')
        plt.legend()
        plt.show()

    return {
        'best_bic': np.argmin(bics) + n_range[0],
        'best_aic': np.argmin(aics) + n_range[0],
        'bic_scores': bics,
        'aic_scores': aics
    }

def select_spectral_k(data: np.ndarray, k_range=(2, 11), plot=True) -> int:
    """
    Determine optimal number of clusters for Spectral Clustering using silhouette score.

    Args:
        data: 2D array-like data for clustering.
        k_range: Range of cluster counts to evaluate.
        plot: Whether to plot silhouette scores.

    Returns:
        Optimal number of clusters (int).
    """
    silhouette_scores = []
    for k in range(*k_range):
        model = SpectralClustering(n_clusters=k, affinity='nearest_neighbors')
        labels = model.fit_predict(data)
        silhouette_scores.append(silhouette_score(data, labels))

    if plot:
        plt.plot(range(*k_range), silhouette_scores, marker='o')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.title('Spectral Clustering Silhouette Scores')
        plt.show()

    return np.argmax(silhouette_scores) + k_range[0]

def estimate_meanshift_bandwidth(data: np.ndarray, quantile=0.2) -> float:
    """
    Estimate optimal bandwidth for Mean Shift clustering.

    Args:
        data: 2D array-like data for clustering.
        quantile: Quantile used for bandwidth estimation.

    Returns:
        Estimated bandwidth (float).
    """
    return estimate_bandwidth(data, quantile=quantile)

def select_birch_k(data: np.ndarray, k_range=(2, 11), plot=True) -> int:
    """
    Determine optimal number of clusters for BIRCH using silhouette score.

    Args:
        data: 2D array-like data for clustering.
        k_range: Range of cluster counts to evaluate.
        plot: Whether to plot silhouette scores.

    Returns:
        Optimal number of clusters (int).
    """
    silhouette_scores = []
    for k in range(*k_range):
        model = Birch(n_clusters=k)
        labels = model.fit_predict(data)
        silhouette_scores.append(silhouette_score(data, labels))

    if plot:
        plt.plot(range(*k_range), silhouette_scores, marker='o')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.title('BIRCH Clustering Silhouette Scores')
        plt.show()

    return np.argmax(silhouette_scores) + k_range[0]

def select_affinity_propagation(data: np.ndarray, preference_values=None, damping=0.9, max_clusters=10) -> float:
    """
    Select optimal preference value for Affinity Propagation yielding <= max_clusters.

    Args:
        data: 2D array-like data for clustering.
        preference_values: Sequence of values to try (optional).
        damping: Damping factor for AffinityPropagation.
        max_clusters: Maximum desired number of clusters.

    Returns:
        Preference value (float) producing an acceptable number of clusters.
    """
    if preference_values is None:
        sim_matrix = -np.linalg.norm(data[:, np.newaxis] - data[np.newaxis, :], axis=2)**2
        preference_values = np.linspace(np.min(sim_matrix), np.median(sim_matrix), 10)

    for preference in preference_values:
        model = AffinityPropagation(preference=preference, damping=damping)
        model.fit(data)
        n_clusters = len(np.unique(model.labels_))
        if n_clusters <= max_clusters:
            return preference

    return preference_values[-1]  # fallback to last value if none under threshold
