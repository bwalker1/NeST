import pandas as pd
import numpy as np
from itertools import combinations, chain
from collections import Counter
import time
from multiprocessing import Pool

from scipy.sparse import csr_matrix, coo_matrix

import scanpy as sc
from anndata import AnnData
from tqdm import tqdm
import networkx as nx

import seaborn as sns
import colorcet as cc
import matplotlib.path
from matplotlib import colors

import warnings

from typing import Optional, Iterable

from .hotspot import get_all_hotspot_keys, compute_hotspot_boundary


def coexpression_hotspots(adata: AnnData, jaccard_matrix=None, hotspot_lookup=None, threshold: float = 0.7,
                          min_size: int = 0, min_genes: int = 5, cutoff: Optional[float] = 0.5, verbose: bool = False,
                          processes: int = 1, divisions: int = 10, use_core: bool = True, core_k: int = 3, 
                          resolution: float = 1.0) -> None:
    """
    Compute coexpression hotspots in a dataset for which some set of single-gene or interaction hotspots have already
    been computed.
    :param adata: AnnData object containing single-gene and/or CCI hotspots previously computed with NeST
    :param jaccard_matrix: Optional, jaccard_matrix previously computed with nest.hotspot_jaccard function.
        Provide together with hotspot_lookup or omit to have them computed from scratch automatically.
    :param hotspot_lookup: Optional, output from nest.hotspot_jaccard, see parameter `jaccard_matrix`.
    :param threshold: Minimum similarity threshold between hotspots to create edge in similarity network
    :param min_size: Minimum number of spots/cells in a coexpression hotspot
    :param min_genes: Minimum number of genes (or CCI hotspots) to combine into a coexpression hotspot
    :param cutoff: Controls how coexpression hotspots are formed from constituent hotspots. Spots in at least this fraction
        of constituent hotspots are in the coexpression hotspot.
    :param verbose: If true, progress information is displayed.
    :param processes: Number of parallel processes to use in computation
    :param divisions: Controls spatial chunking. Larger values will cause less memory usage per process.
    :param use_core: Filter out k-core of similarity network
    :param core_k: k-value to use in k-core (see `use_core`)
    :param resolution: Resolution parameter to use in Leiden clustering algorithm of similarity network.
    """
    if jaccard_matrix is None:
        if verbose:
            print("Computing hotspot similarity matrix.")
        jaccard_matrix, hotspot_lookup = hotspot_jaccard(adata, processes=processes,
                                                         verbose=verbose, divisions=divisions)

    try:
        multigene_hotspots = hotspot_groups(jaccard_matrix, hotspot_lookup, use_core=use_core, core_k=core_k,
                                            resolution=resolution,
                                            threshold=threshold, min_genes=min_genes)
    except RuntimeError as e:
        warnings.warn("No coexpression above threshold was detected. Try again with a lower threshold.")
        return

    median_sizes = []
    for idx, multigene_hotspot in enumerate(multigene_hotspots):
        sizes = []
        for key, v in multigene_hotspot:
            hotspot_size = np.count_nonzero(adata.obs[key] == v)
            sizes.append(hotspot_size)
        median_sizes.append(int(np.median(sizes)))
    multi_arrays = {}
    new_multigene_hotspots = []
    score_dict = {}
    for idx, (multigene_hotspot, median_size) in enumerate(
            sorted(zip(multigene_hotspots, median_sizes), key=lambda x: -x[1])):
        if median_size < min_size:
            break
        hotspots_per_spot = np.zeros(adata.shape[0])

        for idx2, (key, v) in enumerate(multigene_hotspot):
            hotspots_per_spot += (adata.obs[key] == v).astype(np.int_)

        if cutoff is None:
            inds = np.argsort(hotspots_per_spot)[::-1][:median_size]
        else:
            inds = np.where(hotspots_per_spot > cutoff * len(multigene_hotspot))[0]

        score = np.sum(hotspots_per_spot[inds]) / np.sum(hotspots_per_spot)
        score_dict[idx] = score
        # compute essence of overlap

        v = np.full(adata.shape[0], np.nan)
        v[inds] = 0
        multi_arrays[idx] = pd.Categorical(v)
        new_multigene_hotspots.append(multigene_hotspot)

    multi_arrays = {f"hotspots_multi_{idx}": v for idx, (k, v) in enumerate(multi_arrays.items())}
    # adata.obs = pd.concat([adata.obs, pd.DataFrame(multi_arrays, index=adata.obs.index)], axis=1)
    adata.obs.drop(adata.obs.filter(regex='hotspots_multi').columns.tolist(), axis=1, inplace=True)
    tmp_df = adata.obs.merge(pd.DataFrame(multi_arrays, index=adata.obs.index), how="outer",
                             left_index=True,
                             right_index=True, suffixes=('_old', None))
    tmp_df.drop(tmp_df.filter(regex='_old$').columns.tolist(), axis=1, inplace=True)
    adata.obs = tmp_df
    new_multigene_hotspots = {str(idx): v for idx, v in enumerate(new_multigene_hotspots)}
    adata.uns["multi_hotspots"] = new_multigene_hotspots

    # Save standard colors for each coexpression hotspot
    cm1 = sns.color_palette(cc.glasbey_bw, n_colors=len(adata.uns['multi_hotspots']))
    adata.uns['multi_hotspots_colors'] = {k: colors.to_hex(cm1[int(k)]) for k in adata.uns['multi_hotspots'].keys()}

    return


def multi_hotspot_groupings(adata):
    multi_hotspots = [v for v in adata.obs if "hotspots_multi" in v]
    multi_hotspots_df = sc.get.obs_df(adata, keys=multi_hotspots)
    # compute to matrix, 0 -> 1, nan -> 0
    multi_hotspots_array = np.nan_to_num(np.array(multi_hotspots_df, dtype=np.float32) + 1).astype(
        np.int_)
    type_counter = Counter()
    for idx in range(adata.shape[0]):
        t = np.where(multi_hotspots_array[idx, :])
        type_counter[t] += 1

    return sorted(type_counter.items(), key=lambda x: -x[1])


def hotspot_decomposition(adata: AnnData, neg_weight: float = 1.0, verbose=False):
    """
    Decompose the single-gene hotspots computed for each gene in terms of coexpression hotspots 
    by a simple greedy optimization algorithm.

    :param adata: anndata object containing single-gene and coexpression hotspots
    :type adata: AnnData
    :param neg_weight: How much to penalize spots in coex. hotspot not in single gene hotspot (default 1.0)
    :type neg_weight: float
    :param verbose: Controls whether a progress bar is displayed (default: False)
    :type verbose: bool

    :returns: weights_dict dictionary, where keys are genes present in `adata`, and items are a tuple of a binary membership
        vector over all coex. hotspots, and a quality score.
    """
    # assumes that single-gene hotspots and multi-gene hotspots have already been computed
    all_hotspots = [v for v in adata.obs if "hotspots_" in v and "multi" not in v]
    multi_hotspots = [v for v in adata.obs if "hotspots_multi" in v]

    multi_hotspots_df = sc.get.obs_df(adata, keys=multi_hotspots)
    # compute to matrix, 0 -> 1, nan -> 0
    multi_hotspots_array = np.nan_to_num(np.array(multi_hotspots_df, dtype=np.float32) + 1).astype(
        np.int_)

    weights_dict = hotspot_decomposition_keys(adata, all_hotspots, multi_hotspots_array,
                                              neg_weight=neg_weight, verbose=verbose)

    type_counter = Counter()
    if compute_type_counts:
        types = [tuple(np.where(weights)[0]) for (weights, _) in weights_dict.values() if
                 np.sum(weights) > 0]
        for t in types:
            for subset in powerset(t):
                type_counter[subset] += 1
        return weights_dict, sorted(type_counter.items(), key=lambda x: -x[1])

    return weights_dict


def hotspot_decomposition_keys(adata, keys, multi_hotspots_array, neg_weight=None, verbose=False):
    weights_dict = {}
    
    if verbose:
        pbar = tqdm(keys)
    else:
        pbar = keys

    for hotspot in pbar:
        hotspot_vector = np.array(adata.obs[hotspot], dtype=np.float32)
        hotspot_vector[pd.notnull(hotspot_vector)] = 1
        hotspot_vector = np.nan_to_num(np.array(hotspot_vector, dtype=np.float32)).astype(np.int_)

        weights, score = hotspot_decomposition_sub(hotspot_vector, multi_hotspots_array,
                                                   neg_weight=neg_weight)
        hotspot_name = hotspot[9:]
        weights_dict[hotspot_name] = (weights, score)

    return weights_dict


def hotspot_decomposition_sub(hotspot_vector, multi_hotspots_array, neg_weight=None):
    if neg_weight is None:
        neg_weight = 1.0
    original_size = np.sum(hotspot_vector.flatten()).astype(np.float64)
    weights = np.zeros(multi_hotspots_array.shape[1])
    score = 0.0

    while True:
        overlap = np.sum(np.multiply(hotspot_vector.reshape(-1, 1), multi_hotspots_array), axis=0)
        neg_overlap = np.sum(np.multiply(1 - hotspot_vector.reshape(-1, 1), multi_hotspots_array),
                             axis=0)
        score_change = overlap - neg_weight * neg_overlap
        max_ind = np.argmax(score_change)
        if score_change[max_ind] > 0:
            weights[max_ind] = 1
            intersects = np.logical_and(hotspot_vector, multi_hotspots_array[:, max_ind])
            hotspot_vector[intersects] = 0
            score += score_change[max_ind]
        else:
            break

    score /= original_size
    # check if we met the minimum coverage
    # if np.sum(hotspot_vector.flatten())/original_size < min_covered_frac:
    # we didn't so return nothing
    #    return np.zeros(multi_hotspots_array.shape[1]), 0.0
    return weights, score


def hotspot_jaccard(adata, keys=None, processes=4, verbose=False, divisions=5):
    if keys is None:
        keys = get_all_hotspot_keys(adata)

    # Construct a sparse binary assignment matrix representing which spots are members of which hotspots in
    # CSR format
    num_hotspots = 0
    mean_size = 0
    indptr_counter = 0
    offsets = {}

    indices_list = []
    indptr = [0]

    hotspot_sizes = []

    hotspot_lookup = []

    for idx, hotspot_key in enumerate(keys):
        offsets[hotspot_key] = num_hotspots
        v = pd.Categorical(adata.obs[hotspot_key])
        n_hotspots = len(v.categories)
        for c in range(n_hotspots):
            inds = np.where(v.codes == c)[0]
            indices_list.append(inds)
            indptr_counter += len(inds)
            indptr.append(indptr_counter)
            hotspot_sizes.append(len(inds))
            mean_size += len(inds)
            hotspot_lookup.append((hotspot_key, v.categories[c]))

        num_hotspots += n_hotspots

    indices = np.concatenate(indices_list)
    data = np.ones(indices.shape)
    hotspot_matrix = csr_matrix((data, indices, indptr)).tocsc()
    hotspot_matrix.resize(hotspot_matrix.shape[0], adata.shape[0])

    hotspot_sizes = np.array(hotspot_sizes, dtype=np.float32)

    # Iterate over columns, each representing a spot and the hotspots it belongs to
    num_pairs = 0
    combination_iterators = []
    num_pairs_arr = []
    for col in range(hotspot_matrix.shape[1]):
        hotspots = hotspot_matrix[:, col].nonzero()[0]
        l = len(hotspots)
        num_pairs += l * (l - 1) // 2
        num_pairs_arr.append(l * (l - 1) // 2)
        # Append to the running list an iterator that iterates over every pair of hotspots that share the current
        # element
        combination_iterators.append(combinations(hotspots, 2))

    # TODO: maybe handle the case we can't import multiprocessing (does that happen?)
    if processes is not None:
        p = Pool(processes)
    # Spatially divide the spots
    coords = adata.obsm['spatial']
    xmin, ymin = np.min(coords, axis=0)
    xmax, ymax = np.max(coords, axis=0)
    xmax += 0.00001
    ymax += 0.00001
    xdiff = xmax - xmin
    ydiff = ymax - ymin
    # Partition up all the spots based on a grid
    data_list = []
    for idx in range(divisions):
        for idy in range(divisions):
            cur_xmin = xmin + (idx / divisions) * xdiff
            cur_xmax = xmin + ((idx + 1) / divisions) * xdiff

            cur_ymin = ymin + (idy / divisions) * ydiff
            cur_ymax = ymin + ((idy + 1) / divisions) * ydiff
            in_current_element = np.all([coords[:, 0] >= cur_xmin, coords[:, 0] < cur_xmax,
                                         coords[:, 1] >= cur_ymin, coords[:, 1] < cur_ymax],
                                        axis=0)
            inds = np.where(in_current_element)[0]
            if len(inds) == 0:
                continue
            cur_iters = []
            cur_num_pairs = 0
            for ind in inds:
                cur_iters.append(combination_iterators[ind])
                cur_num_pairs += num_pairs_arr[ind]
            data_list.append((cur_iters, cur_num_pairs, num_hotspots))

    num_tasks = len(data_list)

    overlap_matrix = csr_matrix((num_hotspots, num_hotspots))

    if verbose:
        list_wrapper = tqdm
    else:
        list_wrapper = lambda x, total: x

    if processes is not None:
        for sub_matrix in list_wrapper(p.imap_unordered(_multi_wrapper, data_list),
                                       total=num_tasks):
            overlap_matrix += sub_matrix
            del sub_matrix
    else:
        for sub_matrix in list_wrapper((_multi_wrapper(data) for data in data_list),
                                       total=num_tasks):
            overlap_matrix += sub_matrix
            del sub_matrix

    if processes is not None:
        p.close()

    del hotspot_matrix
    overlap_matrix = overlap_matrix.tocoo()

    # Compute the jaccard matrix
    if verbose:
        print("Computing jaccard matrix")
    row_hotspot_size = hotspot_sizes[overlap_matrix.row]
    col_hotspot_size = hotspot_sizes[overlap_matrix.col]
    overlap_count = overlap_matrix.data
    jaccard_matrix_data = row_hotspot_size + col_hotspot_size
    jaccard_matrix_data -= overlap_count
    jaccard_matrix_data = overlap_count / jaccard_matrix_data
    if verbose:
        print("Converting jaccard matrix")
    jaccard_matrix = coo_matrix(
        (jaccard_matrix_data, (overlap_matrix.row, overlap_matrix.col)),
        overlap_matrix.shape).tocsr()

    return jaccard_matrix, hotspot_lookup


def _multi_wrapper(data):
    iters, num_pairs, num_hotspots = data
    return compute_overlap_matrix(iters, num_pairs, num_hotspots, verbose=False)


def compute_overlap_matrix(iterator_list, num_pairs, num_hotspots, verbose=True):
    t0 = time.time()
    idx = 0
    if verbose:
        print(f"Constructing overlap matrix with {num_pairs} pairs.")
    row = np.empty(num_pairs, dtype=np.int32)
    col = np.empty(num_pairs, dtype=np.int32)
    if verbose:
        iterator_list = tqdm(iterator_list)
    for it in iterator_list:
        for i, j in it:
            row[idx], col[idx] = i, j
            idx += 1
    data = np.ones(num_pairs, dtype=np.float32)
    if verbose:
        print("Converting overlap matrix")
    overlap_matrix = coo_matrix((data, (row, col)), (num_hotspots, num_hotspots),
                                dtype=np.float32).tocsr()
    del data, row, col
    if verbose:
        t1 = time.time()
        print(f"Completed in {t1 - t0} seconds.")
    return overlap_matrix


def hotspot_groups(jaccard_matrix, hotspot_lookup, threshold, min_genes=5, use_core=True, core_k=3,
                   resolution=1.0):
    jaccard_matrix_coo = jaccard_matrix.tocoo()
    above_threshold = jaccard_matrix_coo.data > threshold

    if np.count_nonzero(above_threshold) == 0:
        raise RuntimeError("No Jaccard similarity values detected above threshold.")
    row = jaccard_matrix_coo.row[above_threshold]
    col = jaccard_matrix_coo.col[above_threshold]
    data = jaccard_matrix_coo.data[above_threshold]
    jaccard_matrix_coo.data = data
    jaccard_matrix_coo.row = row
    jaccard_matrix_coo.col = col
    jaccard_matrix_reduced = jaccard_matrix_coo.tocsr()
    del jaccard_matrix_coo


    g = nx.from_scipy_sparse_matrix(jaccard_matrix_reduced)
    if use_core:
        g = nx.k_core(g, k=core_k)

    if g.number_of_edges() == 0:
        raise RuntimeError("Not enough Jaccard similarity values detected above threshold.")

    multigene_hotspots = []
    vl = nx.algorithms.community.louvain_communities(g, resolution=resolution)
    for c in sorted(vl, key=lambda x: -len(x)):
        if len(c) < min_genes:
            continue
        multigene_hotspot = [hotspot_lookup[i] for i in c]
        multigene_hotspots.append(multigene_hotspot)

    return multigene_hotspots


def compute_multi_boundaries(adata, alpha_max, alpha_min, ids=None, verbose=False):
    multi_hotspots = [v for v in adata.obs if "hotspots_multi" in v]

    adata.uns["multi_boundaries"] = {}
    pbar = enumerate(multi_hotspots)
    if verbose:
        pbar = tqdm(pbar, total=len(multi_hotspots))
    for idx, hotspot_key in pbar:
        if ids is not None and idx not in ids:
            continue
        boundary = compute_hotspot_boundary(adata, hotspot_key, 0, alpha_max=alpha_max,
                                            alpha_min=alpha_min)
        # saving of anndata converts dictionary indices in .uns to strings anyway
        adata.uns["multi_boundaries"][str(idx)] = boundary


def multi_closure(adata):
    hotspot_keys = [v for v in adata.obs if "hotspots_multi" in v]
    points = adata.obsm['spatial']
    for idx, hotspot_key in enumerate(hotspot_keys):
        boundary = adata.uns["multi_boundaries"][str(idx)]
        path = matplotlib.path.Path(boundary)
        v = pd.Categorical(path.contains_points(points, radius=-0.01), categories=(True,))
        v.rename_categories([0])
        adata.obs[hotspot_key] = v


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))


def multiset_index(adata: AnnData, 
                   key: str = 'coex_multiset',
                   coex_to_remove: Optional[Iterable] = None) -> None:
    """
    Partition cells/spots based on what coexpression hotspots they're in
    :param adata: Anndata object
    :param key: name of key to add to adata.obs (default: 'coex_multiset')
    """
    obs_names = list(adata.obs)
    activity_names = [v for v in obs_names if "hotspots_multi_" in v]
    multi_df = sc.get.obs_df(adata, keys=activity_names)
    multi_df_arr = pd.notnull(multi_df).to_numpy()
    mapping = {(): 0}
    cc = Counter()
    res = np.empty(multi_df_arr.shape[0], dtype=np.int_)
    count = 1

    for idx in range(multi_df_arr.shape[0]):
        v = tuple(set(np.where(multi_df_arr[idx, :])[0]) - coex_to_remove)
        if v not in mapping:
            mapping[v] = count
            count += 1
        cc[v] += 1
        res[idx] = mapping[v]
    min_size = 50
    for k, v in cc.items():
        if v < min_size:
            res[res == mapping[k]] = 0
    adata.obs['coex_multiset'] = pd.Categorical(res)

    return mapping