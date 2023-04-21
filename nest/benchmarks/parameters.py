"""
Test the sensitivity of nest results on input parameters
"""

import numpy as np
import scanpy as sc
import pandas as pd
import os
import matplotlib.pyplot as plt

import random

import nest
from anndata import AnnData
from typing import Optional, Tuple
from numpy.typing import ArrayLike





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

    # Precompute reference parameter versions, allowing for reuse later
    if 1:
        adata_unsmoothed = adata.copy()
        adata_smoothed = adata.copy()
        nest.spatial_smoothing(adata_smoothed)

        adata_single_gene_hotspots = adata_smoothed.copy()
        nest.compute_gene_hotspots(adata_single_gene_hotspots, verbose=True, log=use_modified_otsu_scaling_v[0],
                                                eps=neighbor_eps_v[0], min_samples=min_samples_v[0], min_size=hotspot_min_size)
        
        adata_ref = adata_single_gene_hotspots.copy()
        nest.coexpression_hotspots(adata_ref, threshold=jaccard_threshold_v[0], min_size=30, cutoff=jaccard_cutoff, min_genes=min_genes,
                                    resolution=resolution_v[0])
        
        
        nest.compute_multi_boundaries(adata, 0.005, 0.00001, verbose=False)

        fig, ax = plt.subplots()
        nest.plot.multi_hotspots(adata, ax=ax, show=False)

        fig.savefig(os.path.join(image_save_dir, f"multi_hotspots_ref.png"), dpi=300, bbox_inches='tight', transparent=True)



    c = 0
    param_to_vary = 0
    for param_to_vary in range(6):
        cur_param_idx = 0 # range(len(params_list[param_to_vary]))
        for cur_param_idx in range(1, len(params_list[param_to_vary])):
            cur_params_list = [params_list[k][cur_param_idx] if k == param_to_vary else params_list[k][0] for k in range(len(params_list))]

            use_smoothing, use_modified_otsu_scaling, neighbor_eps, min_samples, jaccard_threshold, jaccard_resolution = cur_params_list
            
            # rescale min_samples
            min_samples = np.ceil(min_samples * (neighbor_eps/neighbor_eps_v[0])**2).astype(np.int_)

            print(c, use_smoothing, use_modified_otsu_scaling, neighbor_eps, min_samples, jaccard_threshold, jaccard_resolution )
            c += 1

            if param_to_vary < 4:
                if use_smoothing:
                    adata = adata_smoothed.copy()
                else:
                    adata = adata_unsmoothed.copy()

                res = nest.compute_gene_hotspots(adata, verbose=False, log=use_modified_otsu_scaling,
                                                eps=neighbor_eps, min_samples=min_samples, min_size=hotspot_min_size)
            else:
                adata = adata_single_gene_hotspots.copy()

            
            jaccard_matrix, hotspot_lookup = nest.hotspot_jaccard(adata, processes=10,
                                                                verbose=False, divisions=10)
            
            nest.coexpression_hotspots(adata, threshold=jaccard_threshold, min_size=30, cutoff=jaccard_cutoff, min_genes=min_genes,
                                    resolution=jaccard_resolution)
            try:
                nest.compute_multi_boundaries(adata, 0.005, 0.00001, verbose=False)
            except ValueError:
                continue

            # Compute the similarity over each chosen hotspot

            fig, ax = plt.subplots()
            nest.plot.multi_hotspots(adata, ax=ax, show=False)

            fig.savefig(os.path.join(image_save_dir, f"multi_hotspots_{use_smoothing}_{use_modified_otsu_scaling}_{neighbor_eps}_{min_samples}_{jaccard_threshold}_{jaccard_resolution}.png"), dpi=300, bbox_inches='tight', transparent=True)
