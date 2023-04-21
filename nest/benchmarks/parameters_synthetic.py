"""
Test the sensitivity of nest results on input parameters
"""

import numpy as np
import scanpy as sc
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import jaccard_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment

import random
from collections import defaultdict

import nest
from nest.synthetic.data import create_synthetic_data 
from anndata import AnnData
from typing import Optional, Tuple
from numpy.typing import ArrayLike



def score_similarity(adata):
    layer_matrix = adata.uns['layer_matrix']
    ch_df = sc.get.obs_df(adata, keys=[v for v in adata.obs.columns if "hotspots_multi" in v])
    multi_hotspots = np.array(pd.notnull(ch_df)).astype(np.int_)
    n_pixels = layer_matrix.shape[0]
    
    def compute_sim_vals(arr, ref_denom=None):
        arr = arr.copy().astype(np.float_)
        arr /= (np.sum(arr, axis=1, keepdims=True)+1e-8)
        a1 = arr[:, np.newaxis, :]
        a2 = arr[np.newaxis, :, :]
        v = np.sum(a1*a2, axis=-1)

        d1 = np.diag(v).reshape(-1, 1)
        d2 = np.diag(v).reshape(1, -1)

        denom = (d1+d2+1e-20)
        if ref_denom is not None:
            denom = np.maximum(denom, ref_denom)

        sim = (2*v)/denom

        return sim, denom

    s1, d = compute_sim_vals(layer_matrix)
    s2, _ = compute_sim_vals(multi_hotspots, d)
    
    score = 1 - (np.linalg.norm(s1-s2)**2)/(n_pixels**2)

    return score

def pairwise_jaccard(adata, label=None):
    layer_matrix = adata.uns['layer_matrix']
    if label is None:
        ch_df = sc.get.obs_df(adata, keys=[v for v in adata.obs.columns if "hotspots_multi" in v])
        multi_hotspots = np.array(pd.notnull(ch_df)).astype(np.int_)
    else:
        multi_hotspots = pd.get_dummies(adata.obs[label]).to_numpy().astype(np.int_)
    n_pixels = layer_matrix.shape[0]

    sim_matrix = np.zeros(shape=[layer_matrix.shape[1], multi_hotspots.shape[1]])

    for idx in range(layer_matrix.shape[1]):
        labels_1 = layer_matrix[:, idx]
        for idy in range(multi_hotspots.shape[1]):
            labels_2 = multi_hotspots[:, idy]
            sim_matrix[idx, idy] = jaccard_score(labels_1, labels_2)

    row_ind, col_ind = linear_sum_assignment(1-sim_matrix)

    res = np.zeros(max(layer_matrix.shape[1], multi_hotspots.shape[1]))
    res[:len(row_ind)] = sim_matrix[row_ind, col_ind]
       
    return np.mean(res)  


def seg_similarity(adata, label):
    regions = adata.obs['regions']
    score = normalized_mutual_info_score(regions, adata.obs[label])
    return score



if __name__=="__main__":
    # Tuning parameters
    # First value is default/base, others are variations to test for sensitivity
    # Test varying one parameter at a time (leave others at base)
    neighbor_eps_v = [0.1, 0.02, 0.05, 0.2]
    density_v = [0.2, 0.05, 0.1, 0.4, 0.6, 0.8]
    jaccard_threshold_v = [0.3, 0.15, 0.2, 0.25, 0.35, 0.4, 0.45]
    #neighbor_eps_v = [0.1]
    #density_v = [0.2]
    #jaccard_threshold_v = [0.3]
    #resolution_v = [1.0, 0.0, 0.1, 0.5, 2.0, 5.0, 10.0]
    resolution_v = np.linspace(0.0, 10.0, 20)
    params_list = [neighbor_eps_v, 
                   density_v, jaccard_threshold_v, resolution_v]


    # Non-tuning parameters
    hotspot_min_size = 20
    min_genes = 3


    dataset = "synthetic_hierarchy"

    # Configuration for plotting and saving
    dataset_dir = os.path.expanduser("~/Documents/data/")
    cache_dir = os.path.expanduser(f"data/{dataset}")
    image_save_dir = os.path.expanduser(f"images/{dataset}/")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    if not os.path.exists(image_save_dir):
        os.makedirs(image_save_dir)

    np.random.seed(5)

    for spatial_genes in [32]:
        for trial in range(1):
            adata = create_synthetic_data(n_pixels = 64, n_genes = 2048, spatial_genes = spatial_genes, n_layers = 5, log1p=True, type="hierarchy",
                                            dropout_active = 0.5,
                                            mean_low=1.0, mean_high=5.0)
            

            columns = defaultdict(list)

            # Precompute reference parameter versions, allowing for reuse later
            if 1:
                adata_base = adata.copy()
                adata_single_gene_hotspots = adata.copy()
                min_samples = nest.get_min_samples_from_density(adata, neighbor_eps_v[0], density_v[0])
                res = nest.compute_gene_hotspots(adata_single_gene_hotspots, verbose=True, log=True,
                                                        eps=neighbor_eps_v[0], min_samples=min_samples, min_size=hotspot_min_size)
                adata_ref = adata_single_gene_hotspots.copy()
                #nest.coexpression_hotspots(adata_ref, threshold=0.3, min_size=1, cutoff=0.5, min_genes=3, resolution=1.000)
                nest.coexpression_hotspots(adata_ref, threshold=jaccard_threshold_v[0], min_size=1, cutoff=0.3, min_genes=min_genes,
                                            resolution=resolution_v[0], processes=4)
                
                
                nest.compute_multi_boundaries(adata_ref, 0.005, 0.00001, verbose=False)
                nest.multi_closure(adata_ref)

                score = pairwise_jaccard(adata_ref)

                print(score)

                columns['eps'].append(neighbor_eps_v[0])
                columns['density'].append(density_v[0])
                columns['threshold'].append(jaccard_threshold_v[0])
                columns['resolution'].append(resolution_v[0])
                columns['varied'].append(np.nan)
                columns['score'].append(score)

                hmrf = nest.hmrf.HMRFSegmentationModel(adata, regions=5, k=4, label_name="class_hmrf")
                hmrf.fit(max_iterations=100, verbose=True, update_labels=True)

                jaccard = pairwise_jaccard(adata, label="class_hmrf")
                nmi = seg_similarity(adata, "class_hmrf")
                print(nmi)
                print(jaccard)

                spagcn = nest.methods.SpaGCN(regions=5)
                spagcn.fit(adata, verbose=True)

                jaccard = pairwise_jaccard(adata, label="class_spagcn")
                nmi = seg_similarity(adata, "class_spagcn")
                print(nmi)
                print(jaccard)

                asdf = 1

                #fig, ax = plt.subplots()
                #nest.plot.multi_hotspots(adata_ref, ax=ax, show=False, spot_size=1/128)
                #plt.show()

                #fig.savefig(os.path.join(image_save_dir, f"multi_hotspots_ref.png"), dpi=300, bbox_inches='tight', transparent=True)

            exit(0)

            c = 0
            param_to_vary = 0
            print("Starting parameter variation")
            scores = []
            param_varied_v = []
            
            for param_to_vary in range(len(params_list)):
                for cur_param_idx in range(1, len(params_list[param_to_vary])):
                    try:
                        cur_params_list = [params_list[k][cur_param_idx] if k == param_to_vary else params_list[k][0] for k in range(len(params_list))]

                        neighbor_eps, density, jaccard_threshold, jaccard_resolution = cur_params_list
                        min_samples = nest.get_min_samples_from_density(adata_base, neighbor_eps, density)

                        print(c, neighbor_eps, min_samples, jaccard_threshold, jaccard_resolution )
                        c += 1

                        if param_to_vary < 2:
                            adata = adata_base.copy()
                            res = nest.compute_gene_hotspots(adata, verbose=False, log=True,
                                                            eps=neighbor_eps, min_samples=min_samples, min_size=hotspot_min_size)
                        else:
                            adata = adata_single_gene_hotspots.copy()

                        
                        jaccard_matrix, hotspot_lookup = nest.hotspot_jaccard(adata, processes=4,
                                                                            verbose=False, divisions=10)
                        
                        nest.coexpression_hotspots(adata, threshold=jaccard_threshold, min_size=10, cutoff=0.3, min_genes=min_genes,
                                                resolution=jaccard_resolution)
                        try:
                            nest.compute_multi_boundaries(adata, 0.005, 0.00001, verbose=False)
                        except ValueError:
                            continue

                        nest.multi_closure(adata)

                        # Compute the similarity over each chosen hotspot
                        score = pairwise_jaccard(adata)
                        print("Score is ", score)
                        scores.append(score)
                        #params_v.append(cur_params_list)
                        columns['eps'].append(neighbor_eps)
                        columns['density'].append(density)
                        columns['threshold'].append(jaccard_threshold)
                        columns['resolution'].append(jaccard_resolution)
                        columns['varied'].append(param_to_vary)
                        columns['score'].append(score)
                        #param_varied_v.append(param_to_vary)

                        #fig, ax = plt.subplots()
                        #nest.plot.multi_hotspots(adata, ax=ax, show=False, spot_size=1/128)

                        #fig.savefig(os.path.join(image_save_dir, f"multi_hotspots_{neighbor_eps}_{min_samples}_{jaccard_threshold}_{jaccard_resolution}.png"), dpi=300, bbox_inches='tight', transparent=True)
                        #plt.show()
                    except:
                        continue



            

            df = pd.DataFrame(data=columns)
            df.to_csv(f'data/synthetic_hierarchy/results_{spatial_genes}_{trial}.csv')