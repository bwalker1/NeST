"""
Generate synthetic datasets for use in training with known ground truth data
"""

import numpy as np
import scanpy as sc
import pandas as pd

import random

import nest
from anndata import AnnData
from typing import Optional, Tuple
from numpy.typing import ArrayLike

from scipy.sparse import coo_matrix, csr_matrix


def create_synthetic_data(n_pixels: Tuple[int, Optional[int]], n_genes: int, spatial_genes: int, n_layers: int,
                          mean_low: float, mean_high: float, dropout_active: float,
                          type: str = "layer", log1p: bool = False) -> AnnData:
    """
    Create a anndata object containing synthetically generated genes exhibiting spatial structural expression.

    Options for spatial structure:
    - Layered
    - Splotches (reminiscent of tumor structure)
    
    TODO:
    Add additional patterns:
    - Curved layers
    Share patterns across genes with some sort of co-expression dependencies

    :param n_pixels: Resolution in pixels of output dataset
    :type n_pixels: int or (int, int)
    :param n_genes: Number of genes to generate in dataset
    :type n_genes: int
    :param type: which patterns to include in dataset (by name)
    :type type: str
    :param log1p: Whether or not to apply log1p normalization to output (default True).
    :param bool: bool, optional
    """

    try:
        n_x, n_y = n_pixels
    except TypeError:
        # If resolution is just a single number, the above raises a TypeError, and just use that number as 
        # number of pixels in both dimensions
        n_x, n_y = n_pixels, n_pixels
    n_total_pixels = n_x * n_y

    # Generate grid of coordinates of each pixel
    xvalues = np.linspace(0, 1, n_x)
    yvalues = np.linspace(0, 1, n_y)
    xx, yy = np.meshgrid(xvalues, yvalues)
    points = np.stack([xx.ravel(), yy.ravel()]).T

    var_names = [f"gene_{idx+1}" for idx in range(n_genes)]

    expr_layer = np.zeros(shape=(n_total_pixels, n_genes), dtype=np.float32)

    # Generate the regions 
    activity_matrix = np.zeros(shape=(n_genes, n_layers), dtype=np.int_)
    layer_matrix = np.zeros(shape=(n_total_pixels, n_layers), dtype=np.int_)
    if type == "layer":
        # TODO: fix this to take the n_layers as input
        regions = _get_layer_regions(points, n_layers)

        raise NotImplementedError("Need to compute activity_matrix here now")
    elif type == "hierarchy":
        max_levels = n_layers - 1
        regions = np.zeros(shape = n_total_pixels, dtype=np.int_)
        layer_matrix[:, 0] = 1
        for level in range(max_levels):
            # Split the last layer into 2
            last_layer = np.max(regions)
            if level % 2 == 0:
                # split in x direction
                split_norm = np.array([0, 1]).reshape(2, 1)
            else:
                # split in y direction
                split_norm = np.array([1, 0]).reshape(2, 1)
            last_layer_points = points[regions == last_layer, :]
            split_proj = np.dot(last_layer_points, split_norm)
            split = np.ravel(split_proj > np.median(split_proj)).astype(np.bool_)
            change_inds = np.where(regions == last_layer)[0][split]
            regions[change_inds] = last_layer + 1

            layer_matrix[change_inds, last_layer+1] = 1
            
        n_layers = max_levels + 1
        # figure out which genes are active in which layer
        for idx in range(n_genes):
            if idx < spatial_genes:
                # To create hierarchical structure, all region starting from a certain index are set to 1
                activity_matrix[idx, (idx % n_layers):] = 1
            else:
                pass
    else:
        raise NotImplementedError


    # Compute the simulated gene expression levels
    # For now just do Poisson I guess?
    for idx in range(n_genes):
        activity = activity_matrix[idx, :]

        mean_high = np.random.uniform(low=5.0, high=15.0)
        mean_low = np.random.uniform(low=0.0, high=1.0)
        if np.count_nonzero(activity) > 0:
            dropout_prob = dropout_active
        else:
            # non-spatially expressed genes have high expression everywhere
            activity = np.ones_like(activity)
            dropout_prob = np.random.uniform(0.8, 1.0)
        mean_arr = np.array([mean_low, mean_high])

        mean_vector = mean_arr[activity[regions]]

        expr = np.random.poisson(lam=mean_vector)

        
        expr *= np.random.binomial(p=1-dropout_prob, n=1, size=(n_total_pixels,))

        expr_layer[:, idx] = expr

    adata = AnnData(X=expr_layer, obsm={'spatial': points}, obs={"regions": (regions)},
                    uns={'dims': (n_x, n_y)}, var={'layer': None})
    adata.var_names = var_names

    if log1p:
        sc.pp.log1p(adata)

    adata.uns['layer_matrix'] = layer_matrix

    return adata



def _get_layer_regions(points: ArrayLike) -> np.ndarray:
    # TODO: add hierarchical structure to the layers

    # Generation parameters
    # TODO: make this configurable
    min_layers = 6
    max_layers = 12
    layer_min_width = 0.025
    layer_max_width = 0.4

    kernel_dim = 2

    # Generate the fractional cutoffs between layers (normalized on 0-1)
    n_layers = np.random.randint(low=min_layers, high=max_layers+1)
    print(f"Generating {n_layers} layers")
    widths = np.zeros((n_layers,))
    while np.min(widths) < layer_min_width or np.max(widths) > layer_max_width:
        #widths = np.random.uniform(size=(n_layers))
        widths = np.random.exponential(scale=1.0, size=(n_layers,))
        widths /= np.sum(widths)
        cutoffs = np.cumsum(widths)

    cutoffs *= cutoffs

    # Create high-dimensional positions and layer normal vector
    #extra_coords = np.zeros(shape=(points.shape[0], kernel_dim-2))
    #extra_coords = np.random.standard_normal(size=(points.shape[0], kernel_dim-2))
    #kernel_points = np.concatenate((points, extra_coords), axis=1)
    kernel_points = points

    normal_vec = np.random.standard_normal(kernel_dim)
    normal_vec /= np.linalg.norm(normal_vec)
    normal_vec = normal_vec.reshape((1, 2))

    kernel_matrix = np.random.standard_normal((kernel_dim, kernel_dim))
    kernel_matrix += kernel_matrix.T

    xx = points[:, 0]
    yy = points[:, 1]
    alpha, beta = np.random.uniform(0.5, 2), np.random.uniform(0.5, 2)
    transformed = np.stack([np.power(xx, alpha), np.power(yy, alpha)])
    kernel_prod = (normal_vec @ points.T).ravel()
    #kernel_prod = np.abs(np.einsum('i j, i j -> i', transformed, normal_vec))

    proj_min, proj_max = np.min(kernel_prod), np.max(kernel_prod)
    # rescale cutoffs into the image of the projection
    cutoffs = proj_min + cutoffs * (proj_max - proj_min)
    cutoffs[-1] += 1e9
    # Figure out which layer each point is in by comparing to layer boundary cutoffs
    layer_membership = np.digitize(kernel_prod, cutoffs)

    assert len(np.unique(layer_membership)) == n_layers

    return layer_membership, n_layers


def _get_foreground(pattern: str, xx: ArrayLike, yy: ArrayLike) -> np.ndarray:
    """
    Helper function for randomly generating the foreground pixels (where the gene is active)
    for different types of patterns.

    :param pattern: Name of the pattern to use.
    :type pattern: str
    :param xx: (n_x, n_y) shape array containing x coordinate of each pixel (expressed in pixels)
    :type xx: np.ndarray or arraylike
    :param yy: (n_x, n_y) shape array containing y coordinate of each pixel (expressed in pixels)
    :type yy: np.ndarray or arraylike

    :raises ValueError: If the pattern type is not a valid pattern name.

    :return: A numpy array of size (n_x, n_y) where foreground pixels are set to 1 and background pixels are set to 0.
    :rtype: np.ndarray
    """

    n_x, n_y = xx.shape


    if pattern == "circle":
        # Avoid generating the circle too close to the edge by trimming out a fraction on either side
        # from the pixels that the center can be placed at
        trim = 0.2
        center_x = np.random.uniform(trim, 1-trim)
        center_y = np.random.uniform(trim, 1-trim)

        # Radius is generated as a random fraction of the lesser of n_x and n_y
        radius_frac_min = 0.15
        radius_frac_max = 0.3
        radius = np.random.uniform(radius_frac_min, radius_frac_max)

        res = (xx - center_x)**2 + (yy - center_y)**2 <= radius**2


    elif pattern == "layer":
        # Layer is determined in terms of normal vector (vector of projection), and min and max value on projection
        # Generate unit vector in random direction by generating 2D Gaussian random value and normalizing
        normal_vector = np.random.normal(loc=[0, 0], scale=1, size=(2,))
        normal_vector /= np.linalg.norm(normal_vector)

        # TODO: adjust this generation to make more thin layers
        v_center = np.random.uniform(0.3, 0.7)
        max_width = min(0.2, 0.8*(0.5-np.abs(0.5-v_center)))
        #v_width = np.random.uniform(0.05, 0.2)
        v_width = 0.075
        v_min, v_max = v_center - v_width, v_center + v_width

        points = np.stack([xx.ravel(), yy.ravel()]).T

        v_proj = np.dot(points, normal_vector)
        proj_min, proj_max = np.min(v_proj), np.max(v_proj)

        v_min = proj_min + v_min*(proj_max - proj_min)
        v_max = proj_min + v_max*(proj_max - proj_min)

        #res = np.logical_or(np.abs(v_proj-v_min) < 0.05, np.abs(v_proj-v_max) < 0.05)
        res = np.abs(v_proj - v_center) < 0.1


    else:
        raise ValueError(f"Unkown pattern type: {pattern}. Pattern must be one of ('circle', 'layer')")


    return res

def compute_sim_vals_2(arr, ref_denom=None):
    row = list()
    col = list()
    for ch_idx in range(arr.shape[1]):
        vals = np.where(arr[:, ch_idx])[0]
        for i in range(len(vals)):
            for j in range(i+1, len(vals)):
                row.append(vals[i])
                col.append(vals[j])

    num_pairs = len(row)

    data = np.ones(num_pairs, dtype=np.float32)

    overlap_matrix = coo_matrix((data, (row, col)), (arr.shape[0], arr.shape[0]),
                                dtype=np.float32).tocsr().tocoo()
    
    d = np.sum(arr, axis=1).astype(np.int_)
    d = 1/d



    overlap_matrix.data /= (0.5 * (overlap_matrix.row[d] + overlap_matrix.col[d] + 1e-20))
    
    sim = overlap_matrix.tocsr()

    return sim, d


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


def score_similarity(layer_matrix, multi_hotspots):
    multi_hotspots = np.array(pd.notnull(multi_hotspots)).astype(np.int_)
    n_pixels = layer_matrix.shape[0]

    s1, d = compute_sim_vals(layer_matrix)
    s2, _ = compute_sim_vals(multi_hotspots, d)
    
    score = 1 - (np.linalg.norm(s1-s2)**2)/(n_pixels**2)

    return score


def score_similarity_adata(adata_true, adata_pred):
    assert(np.array_equal(adata_true.shape[0], adata_pred.shape[0]))

    ch_df_true = sc.get.obs_df(adata_true, keys=[v for v in adata_true.obs.columns if "hotspots_multi" in v])
    ch_df_pred = sc.get.obs_df(adata_pred, keys=[v for v in adata_pred.obs.columns if "hotspots_multi" in v])

    mh_true = np.array(pd.notnull(ch_df_true)).astype(np.int_)
    mh_pred = np.array(pd.notnull(ch_df_pred)).astype(np.int_)

    n_spots = adata_true.shape[0]

    s1, d = compute_sim_vals_2(mh_true)
    s2, _ = compute_sim_vals_2(mh_pred, d)

    score = 1 - (np.linalg.norm(s1-s2)**2)/(n_spots**2)

    return score





if __name__=="__main__":
    from timeit import timeit
    from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
    score_fn = adjusted_rand_score
    # Create a synthetic dataset and plot the resulting gene expression
    np.random.seed(5)
    if 1:
        adata, layer_matrix = create_synthetic_data(n_pixels = 64, n_genes = 256, spatial_genes = 256, n_layers = 8, log1p=True, type="hierarchy",
                                      dropout_active = 0.5,
                                      mean_low=1.0, mean_high=5.0)
        print("Generated data")
        show = False
        if show:
            sc.pl.spatial(adata, color=adata.var_names[:8], color_map="Blues", frameon=False,
                        spot_size=1/128)

        from nest.hmrf import HMRFSegmentationModel
        n_layers = len(np.unique(adata.obs['regions']))

        use_hmrf = False
        use_nest = True
        use_spagcn = False
        use_nmf = False

        if use_nmf:
            from sklearn.decomposition import NMF
            try:
                X = adata.X.toarray()
            except AttributeError:
                X = adata.X

            model = NMF(n_components=n_layers, init='random', random_state=0)
            W = model.fit_transform(X)
            H = model.components_

            for idx in range(n_layers):
                adata.obs[f'nmf_{idx}'] = W[:, idx]

            var_names = [f"nmf_{k}" for k in range(n_layers)]
            nest.plot.spatial(adata, color=var_names, cmap="Blues", spot_size=1/128)

        if use_hmrf:
            adata_reduced = adata.copy()
            #sc.pp.highly_variable_genes(adata_reduced, n_top_genes=2000, subset=True)
            hmrf = HMRFSegmentationModel(adata=adata_reduced, regions=5, k=4, label_name="class_hmrf")
            hmrf.fit(max_iterations=200, verbose=True, update_labels=True)

            # Compute score
            score = score_fn(adata_reduced.obs['regions'], adata_reduced.obs['class_hmrf'])
            print(f"Score is {score}")

            sc.pl.spatial(adata_reduced, color=("regions", "class_hmrf"), frameon=False,
                        spot_size=1/128)
            

        if use_spagcn:
            adata_reduced = adata.copy()
            spagcn = nest.methods.SpaGCN(regions=n_layers)
            spagcn.fit(adata_reduced, verbose=True)

            score = score_fn(adata_reduced.obs['regions'], adata_reduced.obs['class_spagcn'])
            print(f"Score is {score}")

            #sc.pl.spatial(adata_reduced, color=("regions", "class_spagcn"), frameon=False,
            #            spot_size=1/128)

            
        if use_nest:
            neighbor_eps = 2*(np.sqrt(2)/64 + 0.01)
            min_samples = nest.get_min_samples_from_density(adata, neighbor_eps, density=0.2)
            hotspot_min_size = 20
            num_hotspot_genes = nest.compute_gene_hotspots(adata, verbose=True, log=True,
                                                           eps=neighbor_eps, min_samples=min_samples, 
                                                           min_size=hotspot_min_size)
            print(f"{num_hotspot_genes} hotspot genes found.")

            #for gene in adata.var_names:
            #    nest.plot.hotspots(adata, color=gene, spot_size=1/128)

            nest.coexpression_hotspots(adata, threshold=0.3, min_size=1, cutoff=0.5, min_genes=3, resolution=10.0)
            num_ch = len(adata.uns['multi_hotspots'])
            #for gene in adata.var_names[:8]:
            #    nest.plot.hotspots(adata, color=gene, spot_size=1/128)
            nest.compute_multi_boundaries(adata, 0.005, 0.00001)
            nest.multi_closure(adata)
            #nest.plot.multi_hotspots(adata, spot_size=1/128)

            # Compute similarity
            ch_df = sc.get.obs_df(adata, keys=[v for v in adata.obs.columns if "hotspots_multi" in v])
            score = score_similarity(layer_matrix, ch_df)


            print(f"Score is {score}")

            if score < 0.99 or True:
                nest.plot.multi_hotspots(adata, spot_size=1/128)


    else:
        print(timeit(lambda: create_synthetic_data(n_pixels = 128, n_genes = 128), number=100)/100)