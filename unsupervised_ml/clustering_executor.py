import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, Birch, AffinityPropagation, MeanShift, estimate_bandwidth, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from collections import Counter
import pandas as pd

def run_all_clustering_algorithms(data, clustering_params):
    """
    Run a suite of clustering algorithms on the given data with provided parameters.

    Parameters:
        data (np.ndarray or pd.DataFrame): The data to cluster.
        clustering_params (dict): A dictionary mapping method names to instantiated sklearn cluster objects.

    Returns:
        dict: A dictionary mapping method names to predicted cluster labels (as strings).
    """
    labels = {}
    for name, method in clustering_params.items():
        if hasattr(method, 'fit_predict'):
            labels[name] = method.fit_predict(data)
        else:
            method.fit(data)
            labels[name] = method.predict(data)
    return {name: np.array(lbls).astype(str) for name, lbls in labels.items()}

def extract_unique_label_column(df, source_col, delimiter=' - '):
    """
    Create a more general group label column based on frequency of values before a delimiter.

    Parameters:
        df (pd.DataFrame): DataFrame containing the source column.
        source_col (str): Column name to parse.
        delimiter (str): Delimiter to split labels on. Default is ' - '.

    Returns:
        pd.Series: A new column with grouped labels (either the prefix or full label).
    """
    parts = df[source_col].astype(str).str.split(delimiter)
    base_labels = parts.str[0]
    label_counts = Counter(base_labels)

    def get_label(row):
        base = row.split(delimiter)[0]
        return row if label_counts[base] > 1 else base

    return df[source_col].apply(get_label)
