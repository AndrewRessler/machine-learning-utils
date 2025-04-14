import numpy as np
from collections import Counter
from scipy.optimize import linear_sum_assignment


def align_clusters(reference, to_align):
    """
    Align cluster labels in to_align to best match those in reference using the Hungarian algorithm.

    Parameters:
        reference (np.ndarray): Reference label array
        to_align (np.ndarray): Label array to align to reference

    Returns:
        np.ndarray: Aligned label array
    """
    ref_labels = np.unique(reference)
    tgt_labels = np.unique(to_align)
    cost = np.zeros((len(ref_labels), len(tgt_labels)))
    for i, ref in enumerate(ref_labels):
        for j, tgt in enumerate(tgt_labels):
            cost[i, j] = -np.sum((reference == ref) & (to_align == tgt))
    row_ind, col_ind = linear_sum_assignment(cost)
    mapping = {tgt_labels[j]: ref_labels[i] for i, j in zip(row_ind, col_ind)}
    return np.array([mapping[label] for label in to_align])


def perform_voting_ensemble(labels_dict, reference_key='KMeans'):
    """
    Perform majority voting ensemble clustering.

    Parameters:
        labels_dict (dict): Dictionary of clustering method -> label arrays
        reference_key (str): Key of the clustering to use as reference for alignment

    Returns:
        np.ndarray: Final ensemble labels
    """
    reference = labels_dict[reference_key]
    aligned = [reference]
    for key, labels in labels_dict.items():
        if key != reference_key:
            aligned_labels = align_clusters(reference, labels)
            aligned.append(aligned_labels)
    majority_labels = []
    for i in range(len(reference)):
        votes = [clustering[i] for clustering in aligned]
        majority = Counter(votes).most_common(1)[0][0]
        majority_labels.append(majority)
    label_set = sorted(set(majority_labels))
    label_map = {label: i for i, label in enumerate(label_set)}
    numeric = np.array([label_map[label] for label in majority_labels])
    return numeric
