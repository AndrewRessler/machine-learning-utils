import numpy as np
import pandas as pd
import plotly.express as px
from collections import Counter
from sklearn.decomposition import PCA, KernelPCA, FastICA
from sklearn.manifold import TSNE, MDS, Isomap
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from plotly.subplots import make_subplots
import umap
import matplotlib.pyplot as plt
import warnings


def create_color_map(labels):
    unique_labels = np.unique(labels)
    return {
        label: px.colors.qualitative.Alphabet[i % len(px.colors.qualitative.Alphabet)]
        for i, label in enumerate(unique_labels)
    }


def plot_clusters_interactive(data_2d, labels, hover_data=None, title='Cluster Visualization'):
    df = pd.DataFrame(data_2d, columns=['Component 1', 'Component 2'])
    df['Cluster'] = labels
    if hover_data is not None:
        for col in hover_data.columns:
            df[col] = hover_data[col].values

    color_map = create_color_map(df['Cluster'])
    return px.scatter(
        df,
        x='Component 1',
        y='Component 2',
        color='Cluster',
        title=title,
        color_discrete_map=color_map,
        hover_data=df.columns if hover_data is not None else ['Component 1', 'Component 2', 'Cluster']
    )


def project_and_plot_all(data, labels_dict, hover_data=None):
    methods = {
        'PCA': PCA(n_components=2),
        't-SNE': TSNE(n_components=2, random_state=0),
        'LDA': 'lda',  # special case
        'Kernel PCA': KernelPCA(n_components=2, kernel='rbf'),
        'ICA': FastICA(n_components=2),
        'MDS': MDS(n_components=2, dissimilarity='euclidean', random_state=42),
        'Isomap': Isomap(n_components=2),
        'UMAP': umap.UMAP(n_components=2, random_state=42)
    }

    cluster_keys = list(labels_dict.keys())

    for method_name, reducer in methods.items():
        fig = make_subplots(rows=2, cols=4, subplot_titles=[f'{method_name} - {name}' for name in cluster_keys])

        for i, name in enumerate(cluster_keys):
            cluster_labels = labels_dict[name]
            if method_name == 'LDA':
                n_classes = len(np.unique(cluster_labels))
                n_components = min(2, n_classes - 1)
                if n_components < 1:
                    continue
                reducer = LDA(n_components=n_components)
                proj = reducer.fit_transform(data, cluster_labels)
                if n_components == 1:
                    proj = np.hstack([proj, np.zeros((proj.shape[0], 1))])
            else:
                proj = reducer.fit_transform(data)

            subfig = plot_clusters_interactive(proj, cluster_labels, hover_data, f'{method_name} - {name}')
            row = i // 4 + 1
            col = i % 4 + 1
            for trace in subfig['data']:
                fig.add_trace(trace, row=row, col=col)

        fig.update_layout(height=600, showlegend=False, title_text=f'{method_name} Cluster Projections')
        fig.show()


# Silence specific warning for TSNE
warnings.filterwarnings("ignore", message="n_jobs value 1 overridden to 1 by setting random_state.")
