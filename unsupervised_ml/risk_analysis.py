import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from collections import Counter
import plotly.graph_objects as go
import plotly.express as px
from sklearn.neighbors import NearestNeighbors
import umap


def calculate_similarity_based_score(matrix, reference_indices, target_index, k=3):
    reference_indices = np.array(reference_indices)
    global_score = np.mean(matrix[reference_indices, target_index])
    reference_profiles = matrix[reference_indices][:, reference_indices]
    target_profile = matrix[reference_indices, target_index].reshape(1, -1)

    k = min(k, len(reference_indices))
    nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(reference_profiles)
    distances, indices = nbrs.kneighbors(target_profile)
    nearest_indices = reference_indices[indices[0]]
    weights = 1 / (distances[0] + 1e-5)
    weighted_score = np.average(matrix[nearest_indices, target_index], weights=weights)
    max_score = np.max(matrix[reference_indices, target_index])
    all_similarities = matrix[reference_indices, :].mean(axis=0)
    rank_score = np.mean(all_similarities > global_score)
    combined = 0.3 * global_score + 0.3 * weighted_score + 0.2 * max_score + 0.2 * rank_score
    return combined


def assign_risk_levels(risk_scores, reference_flags, percentiles=[25, 50, 75, 90]):
    bounds = np.percentile(risk_scores, percentiles)
    levels = []
    for score, is_ref in zip(risk_scores, reference_flags):
        if is_ref:
            levels.append('Reference')
        elif score >= bounds[3]:
            levels.append('Very High Risk')
        elif score >= bounds[2]:
            levels.append('High Risk')
        elif score >= bounds[1]:
            levels.append('Medium Risk')
        elif score >= bounds[0]:
            levels.append('Low Risk')
        else:
            levels.append('Very Low Risk')
    return levels


def plot_umap(df, matrix, label_col, score_col, is_3d=False):
    reducer = umap.UMAP(n_components=3 if is_3d else 2, random_state=42)
    umap_data = reducer.fit_transform(matrix)
    df = df.copy()
    for i in range(umap_data.shape[1]):
        df[f'UMAP{i+1}'] = umap_data[:, i]
    base = px.scatter_3d if is_3d else px.scatter
    fig = base(
        df[df[label_col] != 'Reference'],
        x='UMAP1', y='UMAP2',
        z='UMAP3' if is_3d else None,
        color=label_col,
        hover_data=[score_col],
        color_discrete_map={
            'Very Low Risk': '#32CD32',
            'Low Risk': '#ADFF2F',
            'Medium Risk': '#FFD700',
            'High Risk': '#FFA500',
            'Very High Risk': '#FF4500',
            'Reference': '#FF0000'
        },
        title=f'UMAP {"3D" if is_3d else "2D"} Visualization by Risk Level'
    )
    return fig


def plot_risk_bar_distribution(risk_levels):
    counts = pd.Series(risk_levels).value_counts()[['Reference', 'Very High Risk', 'High Risk', 'Medium Risk', 'Low Risk', 'Very Low Risk']]
    colors = ['#FF0000', '#FF4500', '#FFA500', '#FFD700', '#ADFF2F', '#32CD32']
    plt.figure(figsize=(10, 6))
    bars = plt.bar(counts.index, counts.values, color=colors)
    plt.title('Distribution of Risk Levels')
    plt.xlabel('Risk Level')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height(), str(bar.get_height()), ha='center')
    plt.tight_layout()
    plt.show()


def plot_risk_histogram(scores):
    plt.figure(figsize=(10, 6))
    sns.histplot(scores, kde=True)
    plt.title('Distribution of Risk Scores')
    plt.xlabel('Risk Score')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()


def build_risk_graph(data_df, matrix, label_col, score_col, threshold=0.7):
    G = nx.Graph()
    for _, row in data_df.iterrows():
        G.add_node(row['company'], risk_level=row[label_col], risk_score=row[score_col])
    for i in range(len(data_df)):
        for j in range(i + 1, len(data_df)):
            if matrix[i, j] > threshold:
                G.add_edge(data_df.iloc[i]['company'], data_df.iloc[j]['company'], weight=matrix[i, j])
    return G


def plot_network(G):
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=0.5, color='#888'), hoverinfo='none')
    node_x, node_y, node_color, node_size, node_text = [], [], [], [], []
    scheme = {
        'Reference': '#FF0000', 'Very High Risk': '#FF4500', 'High Risk': '#FFA500',
        'Medium Risk': '#FFD700', 'Low Risk': '#ADFF2F', 'Very Low Risk': '#32CD32'
    }
    for node, data in G.nodes(data=True):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_color.append(scheme[data['risk_level']])
        node_size.append(20 if data['risk_level'] == 'Reference' else 15 if data['risk_level'] == 'Very High Risk' else 10)
        node_text.append(f"{node}<br>Level: {data['risk_level']}<br>Score: {data['risk_score']:.2f}")
    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers', hoverinfo='text', text=node_text,
                            marker=dict(color=node_color, size=node_size, line=dict(width=2)))
    fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(
        title='Network of Entities by Risk Level',
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    for level, color in scheme.items():
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=10, color=color),
                                 legendgroup=level, showlegend=True, name=level))
    fig.show()


def summarize_network(G):
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    levels = nx.get_node_attributes(G, 'risk_level')
    scores = nx.get_node_attributes(G, 'risk_score')
    dist = Counter(levels.values())
    print("\nRisk Level Distribution:")
    for level, count in dist.items():
        print(f"{level}: {count}")
    print("\nAverage Degree by Risk Level:")
    for level in set(levels.values()):
        degs = [G.degree(n) for n in G.nodes if levels[n] == level]
        print(f"{level}: {np.mean(degs):.2f}")
