"""
Test the sensitivity of nest results on input parameters
"""

import numpy as np
import scanpy as sc
import pandas as pd
import os
import matplotlib.pyplot as plt

import random
import seaborn as sns

import nest
from anndata import AnnData
from typing import Optional, Tuple
from numpy.typing import ArrayLike

from nest.synthetic.data import score_similarity_adata
from sklearn.metrics import jaccard_score


def compute_max_jaccard(adata, adata_comp):
    ch_df_true = np.array(pd.notnull(sc.get.obs_df(adata, keys=[v for v in adata.obs.columns if "hotspots_multi" in v]))).astype(np.int_)
    ch_df_comp = np.array(pd.notnull(sc.get.obs_df(adata_comp, keys=[v for v in adata_comp.obs.columns if "hotspots_multi" in v]))).astype(np.int_)

    jaccard_scores = []
    for idx in range(ch_df_true.shape[1]):
        max_jaccard = max([jaccard_score(ch_df_true[:, idx], ch_df_comp[:, idy]) for idy in range(ch_df_comp.shape[1])])
        jaccard_scores.append(max_jaccard)

    return jaccard_scores


if __name__=="__main__":
    # Tuning parameters
    # First value is default/base, others are variations to test for sensitivity
    # Test varying one parameter at a time (leave others at base)
    use_smoothing_v = [True, False]
    use_modified_otsu_scaling_v = [True, False]
    neighbor_eps_v = [75, 50, 100]
    # For sake of simplicity we scale this later on proportional to neighbor_eps**2, the area being considered
    min_samples_v = [5, 3, 4, 8, 10]
    jaccard_threshold_v = [0.35, 0.2, 0.25, 0.3, 0.4, 0.45]
    resolution_v = [1.0, 0.0, 0.5, 1.5, 2.0, 5.0]
    params_list = [use_smoothing_v, use_modified_otsu_scaling_v, neighbor_eps_v, 
                   min_samples_v, jaccard_threshold_v, resolution_v]


    # Non-tuning parameters
    hotspot_min_size = 50
    min_genes = 8
    jaccard_cutoff = 0.5


    dataset = "slideseq"

    # Configuration for plotting and saving
    dataset_dir = os.path.expanduser("~/Documents/data/")
    cache_dir = os.path.expanduser(f"data/{dataset}")
    image_save_dir = os.path.expanduser(f"images/{dataset}/")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    if not os.path.exists(image_save_dir):
        os.makedirs(image_save_dir)
    nest.plot.set_dataset_plot_parameters(dataset)

    adata = nest.data.get_data(dataset, dataset_dir, normalize=True)

    adata_smoothed = adata.copy()
    nest.spatial_smoothing(adata_smoothed)

    trials = 5
    
    frac_genes = 0.8
    n_genes = int(np.round(frac_genes*adata.shape[1]))

    # Compute hotspots on original adata
    if 1:
        adata_full = adata_smoothed.copy()
        nest.compute_gene_hotspots(adata_full, verbose=True, log=use_modified_otsu_scaling_v[0],
                                                    eps=neighbor_eps_v[0], min_samples=min_samples_v[0], min_size=hotspot_min_size)
        
        adata_ref = adata_full.copy()
        nest.coexpression_hotspots(adata_full, threshold=jaccard_threshold_v[0], min_size=30, cutoff=jaccard_cutoff, min_genes=min_genes,
                                    resolution=resolution_v[0])
    
        nest.compute_multi_boundaries(adata_full, 0.005, 0.00001, verbose=False)

    res = dict()

    for trial in range(trials):
        # Subsample genes
        gene_list = random.sample(list(adata.var_names), k=n_genes)
        adata_cur = adata_smoothed[:, gene_list].copy()

        # Run analysis
        nest.compute_gene_hotspots(adata_cur, verbose=True, log=use_modified_otsu_scaling_v[0],
                                                eps=neighbor_eps_v[0], min_samples=min_samples_v[0], min_size=hotspot_min_size)
        
        nest.coexpression_hotspots(adata_cur, threshold=jaccard_threshold_v[0], min_size=30, cutoff=jaccard_cutoff, min_genes=min_genes,
                                    resolution=resolution_v[0])
        
        #nest.compute_multi_boundaries(adata_cur, 0.005, 0.00001, verbose=False)
        #fig, ax = plt.subplots()
        #nest.plot.multi_hotspots(adata_cur, ax=ax, show=False)

        # compute overlap between each pair of hotspots
        jaccard_scores = compute_max_jaccard(adata_full, adata_cur)
        print(jaccard_scores)

        res[f'trial_{trial}'] = jaccard_scores

        #plt.show()


        #print(score_similarity_adata(adata_full, adata_cur))
        
        #fig.savefig(os.path.join(image_save_dir, f"multi_hotspots_ref.png"), dpi=300, bbox_inches='tight', transparent=True)


    data = pd.DataFrame(data=res).to_numpy()
    mean_jaccard = data.mean(axis=1)
    cm = sns.color_palette("Reds", as_cmap=True)
    colors = [cm(x) for x in mean_jaccard]

    fig, ax = plt.subplots(1, 1, figsize=[2, 2])

    nest.plot.multi_hotspots(adata_full, color_type=colors, ax=ax, show=False, show_colorbar=True)
    fig.savefig(os.path.join(image_save_dir, f"subsampling_{frac_genes}.pdf"), dpi=300, transparent=True, bbox_inches="tight")
    plt.show()

