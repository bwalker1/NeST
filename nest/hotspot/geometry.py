import numpy as np
from collections import Counter

from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from nest.hotspot import differential_expression, hotspot_decomposition

import pandas as pd


def geometry_scores(adata):
    # Compute a significance score based on each hotspot and how much stuff is consistent
    ncells = adata.shape[0]
    arr = np.zeros(ncells)
    arr_unique = np.zeros(ncells)
    arr_unique_expr = np.zeros(ncells)
    X = adata.X.toarray()
    var_names = np.array([v.casefold() for v in adata.var_names])
    for idx, v in adata.uns['multi_hotspots'].items():
        # subarr = np.zeros(ncells)
        for key, c in v:
            gene_idx = np.where(var_names == key[9:].casefold())[0]
            sub = (adata.obs[key] == int(c)).to_numpy().astype(np.int_).reshape(-1)
            arr += sub
            arr_unique += sub.astype(np.float_) / adata.obs[key].count()
            expr = X[:, gene_idx].reshape(-1)
            if expr.size == 0:
                continue
            arr_unique_expr[sub.astype(np.bool_)] += expr[sub.astype(np.bool_)] / np.sum(expr)
    # normalize
    arr = np.log(1 + arr)
    arr /= np.max(arr)
    arr_unique /= np.max(arr_unique)
    arr_unique_expr /= np.max(arr_unique_expr)
    score_dict = {"geometry": arr,
                  "geometry_unique": arr_unique,
                  "geometry_unique_expr": arr_unique_expr}
    geometry_score_df = pd.DataFrame(score_dict, index=adata.obs.index)
    adata.obs = pd.concat([adata.obs, geometry_score_df], axis=1)
    return arr, arr_unique, arr_unique_expr


def geometric_markers(adata, inds, min_fc=1, use_decomp=False):
    # find all genes in exactly one of the hotspots
    gene_counts = Counter()

    if use_decomp:
        weights_dict = hotspot_decomposition(adata)

    for ind in inds:
        if use_decomp:
            genes = [k for k, v in weights_dict.items() if v[0][ind] and '_' not in k]
        else:
            genes = [k[9:] for k, v in adata.uns['multi_hotspots'][str(ind)]]
        #print(genes)
        for gene in genes:
            if "_" in gene:
                # Need to skip over any interaction hotspots involved in here
                continue
            gene_counts[gene] += 1

    # unique genes are in exactly one multi hotspot
    #print(gene_counts)
    unique_genes = {k for k, v in gene_counts.items() if v == 1}

    markers = {k: [] for k in inds}
    for ind in inds:
        if use_decomp:
            genes = [k for k, v in weights_dict.items() if v[0][ind] and '_' not in k]
        else:
            genes = [k[9:] for k, v in adata.uns['multi_hotspots'][str(ind)]]
        for gene in genes:
            if gene in unique_genes:
                markers[ind].append(gene)
    #print(markers)
    # Now do DE filtering to find the ones that are actually more highly expressed
    for cur_ind in inds:
        inds_a = np.where(pd.notnull(adata.obs[f'hotspots_multi_{cur_ind}']))[0]
        adata_sub = adata.copy()
        adata_sub.obs = pd.DataFrame(index=adata_sub.obs.index)
        adata_sub = adata_sub[:, markers[cur_ind]]
        gene_set = set(markers[cur_ind])
        gene_fc = np.full(fill_value=0, shape=len(gene_set), dtype=np.float_)
        for other_ind in inds:
            if cur_ind == other_ind:
                continue
            inds_b = np.where(pd.notnull(adata.obs[f'hotspots_multi_{other_ind}']))[0]
            de = differential_expression(adata_sub, inds_a, inds_b, use_raw=False)
            # positively_expressed = de.index[de['log2(fc)'] > min_fc]
            # gene_set = gene_set & set(positively_expressed)
            var = np.array(de['log2(fc)'])
            try:
                gene_fc += var
            except ValueError:
                print(gene_fc.shape, gene_fc)
                print(var.shape, var)
        genes_by_fc = [v[0] for v in sorted(zip(markers[cur_ind], gene_fc), key=lambda x: -x[1])]
        markers[cur_ind] = genes_by_fc

    return markers


def similarity_map(adata, idx, adata_ref=None, ax=None, linewidth=0.5, linecolor="black", title=""):
    if adata_ref is None:
        adata_ref = adata
    ncells = adata.shape[0]
    arr = np.zeros(ncells)
    var_names = np.array([v.casefold() for v in adata.var_names])
    for key, c in adata_ref.uns['multi_hotspots'][str(idx)]:
        try:
            sub = pd.notnull(adata.obs[key]).to_numpy().astype(np.int_).reshape(-1)
            arr += sub
        except KeyError:
            continue
    # normalize
    arr /= len(adata_ref.uns['multi_hotspots'][str(idx)])

    return arr


def sim_linkage(adata, method='single'):
    # Compute pairwise similarity between all clusters
    n_ch = len(adata.uns['multi_hotspots'])
    pairwise_sim = np.zeros(shape=(n_ch, n_ch))
    for idx in adata.uns['multi_hotspots'].keys():
        sim = similarity_map(adata, idx)
        pairwise_sim[int(idx), int(idx)] = 1
        for idy in adata.uns['multi_hotspots'].keys():
            if idx == idy:
                continue
            val = np.mean(sim[pd.notnull(adata.obs[f'hotspots_multi_{idy}'])])
            pairwise_sim[int(idx), int(idy)] += val
            # pairwise_sim[int(idy), int(idx)] += val

    pairwise_sim = np.minimum(pairwise_sim, pairwise_sim.T)
    pairwise_dist = 1 - pairwise_sim
    Z = linkage(squareform(pairwise_dist), method=method)
    return pairwise_sim, Z
