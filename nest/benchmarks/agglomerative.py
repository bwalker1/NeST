"""
Test hierarchical agglomerative clustering algorithms for comparison
"""

import numpy as np
import scanpy as sc
import pandas as pd
import squidpy as sq
import os
import matplotlib.pyplot as plt
import seaborn as sns

import random

import nest
from nest.synthetic.data import create_synthetic_data 
from anndata import AnnData
from typing import Optional, Tuple
from numpy.typing import ArrayLike

from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

def get_groups(model):
    membership_dict = dict()
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        cur_list = []
        for child_idx in merge:
            if child_idx < n_samples:
                cur_list.append(child_idx)
            else:
                cur_list.extend(membership_dict[child_idx - n_samples])
        membership_dict[i] = cur_list

    return membership_dict

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

if __name__ == "__main__":
    dataset = "V1_Breast_Cancer_Block_A_Section_1"

    # Configuration for plotting and saving
    dataset_dir = os.path.expanduser("~/Documents/data/")
    cache_dir = os.path.expanduser(f"data/{dataset}")
    image_save_dir = os.path.expanduser(f"images/{dataset}/")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    if not os.path.exists(image_save_dir):
        os.makedirs(image_save_dir)
    nest.plot.set_dataset_plot_parameters(dataset)

    #adata = sq.datasets.slideseqv2()
    #adata = create_synthetic_data(n_pixels = 64, n_genes = 256, spatial_genes = 256, n_layers = 10, log1p=True, type="hierarchy",
    #                                   dropout_active = 0.5,
    #                                   mean_low=1.0, mean_high=5.0)
    adata = nest.data.get_data(dataset, dataset_dir, normalize=True)
    n_pca = 15
    sc.pp.pca(adata, n_comps=n_pca)

    cluster = AgglomerativeClustering(distance_threshold=None, n_clusters=5, linkage="ward")
    cluster.fit(adata.obsm['X_pca'])

    membership = get_groups(cluster)

    for i in range(adata.shape[0]-2, adata.shape[0]-60, -1):
        tmp = membership[i]
        v = np.zeros(adata.shape[0], dtype=np.int_)
        v[tmp] = 1
        adata.obs['tmp'] = pd.Categorical(v, categories=[1])
        fig, ax = plt.subplots(1, 1, figsize=[1, 1])
        nest.plot.spatial(adata, color="tmp", show=False, ax=ax, title="", frameon=False, legend_loc=None, alpha_img=0.5)
        fig.savefig(os.path.expanduser(f"~/Desktop/images/agglomerative_{adata.shape[0] - 2 - i}.png"), dpi=300, transparent=True, bbox_inches="tight")
        plt.close(fig)


    #adata.obs['tmp'] = pd.Categorical(cluster.labels_)
    #nest.plot.spatial(adata, color="tmp", show=True)

    #plot_dendrogram(cluster, truncate_mode="level", p=3)
    #plt.show()