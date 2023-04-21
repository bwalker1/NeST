from sklearn.preprocessing import normalize
from sklearn.cluster import DBSCAN, OPTICS
from skimage.filters.thresholding import threshold_otsu
from statsmodels.stats.multitest import fdrcorrection
from scipy.stats import mannwhitneyu
from scipy.spatial import Delaunay
try:
    from scipy.sparse import csc_array
except ImportError:
    from scipy.sparse import csc_matrix as csc_array
from sklearn.neighbors import BallTree
from tqdm import tqdm
from shapely import geometry
from shapely.ops import unary_union, polygonize
from multiprocessing import Pool
import matplotlib as mpl

from nest.utility import get_neighbor_adjacency

import pandas as pd
import numpy as np


def compute_gene_hotspots(adata, gene_list=None, verbose=False, return_hotspot_counts=False,
                          log=True, **kwargs):
    if gene_list is None:
        gene_list = adata.var_names

    num_hotspot_genes = 0

    # gene_list = gene_list[:max_genes]
    data = adata[:, gene_list].X
    try:
        data = data.tocsc()
    except AttributeError:
        data = csc_array(data)

    region_dict = {}
    hotspot_counts = {}
    if verbose:
        gene_list_range = tqdm(enumerate(gene_list), total=len(gene_list))
    else:
        gene_list_range = enumerate(gene_list)
    for idx, gene in gene_list_range:
        gene_expression = data[:, [idx]]
        try:
            gene_expression = gene_expression.toarray()
        except AttributeError:
            pass

        cutoff = _compute_cutoff(gene_expression, log=log)
        # cutoff = np.quantile(gene_expression, expression_threshold)
        inds = np.where(gene_expression > cutoff)[0]

        if len(inds) == 0:
            continue

        regions = compute_hotspots(adata, input_data=inds,  # output_key=f"hotspots_{gene}",
                                   output_key=None, return_regions=True,
                                   input_threshold=cutoff, **kwargs)

        if regions is not None:
            num_hotspot_genes += 1
            region_dict[f"hotspots_{gene}"] = regions
            hotspot_counts[gene] = len(regions.categories)

        # if verbose:
        #    print(f"{gene} ({idx}/{len(gene_list)}, {counter} hotspot genes found).")

    hotspots_df = pd.DataFrame(region_dict, index=adata.obs.index)
    adata.obs = pd.concat([adata.obs, hotspots_df], axis=1)

    if return_hotspot_counts:
        return hotspot_counts
    else:
        return num_hotspot_genes


def compute_hotspots(adata, min_samples, eps, input_key=None, output_key=None,
                     input_data=None, return_regions=False,
                     min_size=None,
                     input_threshold=None, core_only=True, method=None):
    if input_data is not None:
        inds = input_data
    else:
        # handle either obs or gene
        try:
            v = adata.obs[input_key]
        except KeyError:
            v = adata[:, input_key].X
            try:
                v = v.toarray()
            except AttributeError:
                pass

        # handle continuous data with provided threshold
        if input_threshold is not None:
            v = v > input_threshold

        inds = np.where(v)[0]

    coords = adata.obsm['spatial']
    X = coords[inds]
    if X.shape[0] < min_size:
        return None

    if method is None or method == "DBSCAN":
        clusters = DBSCAN(min_samples=min_samples, eps=eps)
    elif method == "OPTICS":
        clusters = OPTICS(min_samples=min_samples, max_eps=eps, cluster_method="xi")
    else:
        raise NotImplementedError
    clusters.fit(X)
    labels = clusters.labels_

    if core_only:
        labels_ = -1 * np.ones(labels.shape)
        labels_[clusters.core_sample_indices_] = labels[clusters.core_sample_indices_]
        labels = labels_

    # default fill of NA to catch all the points that are not active
    # we can't use np.nan yet because that breaks np.unique
    regions = np.zeros(shape=len(adata), dtype=np.int_)

    # relabel any points whose cluster is not large enough to 0 (noise)
    if min_size is not None:
        vals, counts = np.unique(labels, return_counts=True)
        vals_under_size = vals[counts < min_size]
        for v in vals_under_size:
            labels[labels == v] = -1

    counter = 1
    for c in np.unique(labels):
        size = np.count_nonzero(labels == c)
        this_inds = inds[labels == c]
        if c == -1 or size < min_size:
            # it just stays as is so do nothing
            continue
        else:
            regions[this_inds] = counter
            counter += 1

    n_regions = np.max(regions)
    if n_regions > 0:
        regions = pd.Categorical(regions,
                                 categories=np.arange(1, n_regions + 1)).remove_unused_categories()
        if output_key is not None:
            adata.obs[output_key] = regions
    else:
        regions = None

    if return_regions:
        return regions
    else:
        return n_regions


def hotspot_marker_genes(adata, eps, verbose=False, alpha=0.01, exclude_interaction=True,
                         neighborhood=True, interactions=None):
    """
    For each hotspot, identify genes that are differentially expressed against baselines of:
        1. other spatially nearby (but inactive cells)
        2. other cells of same interaction
    """

    filtered_interactions = adata.uns["interactions"]
    A = get_neighbor_adjacency(adata.obsm['spatial'], eps=eps)

    out = {v: {} for v in filtered_interactions["interaction_name"]}

    cutoff = -np.log10(alpha)

    for _, row in filtered_interactions.iterrows():
        interaction = row["interaction_name"]
        if interactions is not None and interaction not in interactions:
            continue

        ligand, receptor = row["ligand"], row["receptor"]

        # Go over all hotspots for this interaction
        try:
            c = adata.obs["hotspots_%s" % interaction]
        except KeyError:
            continue
        active_array = c.notnull()
        for v in np.unique(c):
            if np.isnan(v):
                continue
            hotspot_array = adata.obs[f"hotspots_{interaction}"] == v
            neighbor_array = np.logical_and(A.dot(np.array(hotspot_array, dtype=np.float64)) > 0,
                                            np.logical_not(hotspot_array))
            hotspot_inds = np.where(hotspot_array)[0]
            neighbor_inds = np.where(neighbor_array)[0]
            other_active_inds = \
                np.where(np.logical_and(active_array, np.logical_not(hotspot_array)))[0]

            # test differential expression against both
            # comparing hotspot to all active
            if np.min([len(hotspot_inds), len(other_active_inds), len(neighbor_inds)]) < 10:
                # very few cells - skip
                continue

            reg1 = differential_expression(adata, hotspot_inds, other_active_inds)
            significant = reg1["-log10(p)"] > cutoff
            reg2 = differential_expression(adata, hotspot_inds, neighbor_inds, alpha=alpha)

            # comparing hotspot to neighboring non-active
            if neighborhood:
                # identify genes that passed both significance tests
                significant = np.logical_and(significant, reg2["-log10(p)"] > cutoff)
                # check that they are differentially expressed the same way
                same_sign = np.sign(reg1["log2(fc)"]) == np.sign(reg2["log2(fc)"])

                selected = np.logical_and(significant, same_sign)
            else:
                selected = significant

            # filter out the interaction genes themsleves
            if exclude_interaction:
                selected = np.logical_and(selected,
                                          np.logical_not(reg1.index.isin([ligand, receptor])))

            # return a list of genes, sorted by magnitude of fold change
            genes = reg1.index[selected]
            avg_fc = (reg1["log2(fc)"] + reg2["log2(fc)"]) / 2
            avg_fc = avg_fc[selected]
            inds = np.argsort(-np.abs(avg_fc))
            genes = genes[inds]
            avg_fc = avg_fc[inds]

            out[interaction][v] = list(zip(genes, avg_fc))

    return out


def differential_expression(adata, inds_a, inds_b, alpha=0.001, max_fc=10, use_raw=True):
    if len(inds_a) == 0 or len(inds_b) == 0:
        raise ValueError("Indices list must be non-empty")

    if adata.raw is not None and use_raw:
        adata_source = adata.raw
    else:
        adata_source = adata
    array_a = adata_source[inds_a, :].X.toarray()
    array_b = adata_source[inds_b, :].X.toarray()

    # compute p-values
    U, pvalue = mannwhitneyu(array_a, array_b)
    _, pvalue = fdrcorrection(pvalue, alpha=alpha)
    pvalue = -np.log10(pvalue)

    # compute fold change
    x_med = np.mean(array_a, axis=0)
    y_med = np.mean(array_b, axis=0)
    # coupled with max_fc avoids division by 0
    eps = 1e-3
    x_med[x_med <= eps] = eps
    y_med[y_med <= eps] = eps
    fc = np.log2(x_med / y_med)
    fc[fc > max_fc] = max_fc
    fc[fc < -max_fc] = -max_fc

    return pd.DataFrame({'log2(fc)': fc, '-log10(p)': pvalue}, index=adata_source.var_names)


def gene_regulation(adata):
    res = {}
    filtered_interactions = adata.uns["interactions"]
    cutoffs = adata.uns["activity_significance_cutoff"]
    for i, row in filtered_interactions.iterrows():
        L, R = row["ligand"], row["receptor"]
        data = adata[:, R].X.toarray()
        expressed_inds = (data > np.quantile(data, 0.9)).reshape(-1)
        interaction_name = row["interaction_name"]
        print(interaction_name)
        try:
            v = np.array(adata.obs[f"activity_{interaction_name}"]) > cutoffs[interaction_name]
        except KeyError:
            # possible that an interaction has active cells but no hotspots so it doesn't get a
            # key here
            continue
        active_inds = np.logical_and(expressed_inds, v)
        inactive_inds = np.logical_and(expressed_inds, np.logical_not(v))

        res[interaction_name] = differential_expression(adata, active_inds, inactive_inds)

    return res


def get_all_hotspot_keys(adata):
    return [v for v in adata.obs if "hotspots_" in v]


def smooth(adata, alpha, k=6, threshold=None):
    smoothing_matrix = normalize(get_neighbor_adjacency(adata.obsm['spatial'], k=k), norm='l1',
                                 axis=1)
    if threshold is not None:
        smoothing_matrix = smoothing_matrix.multiply(get_neighbor_adjacency(adata.obsm['spatial'],
                                                                            eps=threshold))
    data = adata.X
    data = (1 - alpha) * data + alpha * (smoothing_matrix @ data)
    adata.X = data


def _compute_cutoff(gene_expression, log=False):
    if log:
        transformed_expr = np.exp(gene_expression) - 1
        cutoff = np.log(1 + threshold_otsu(transformed_expr))
    else:
        cutoff = threshold_otsu(gene_expression)
    return cutoff


def compute_hotspot_boundary(adata, hotspot_key, region, alpha_max=None, alpha_min=0.001):
    if alpha_max is None:
        alpha_max = 10
    v = adata.obs[hotspot_key]

    coords = adata.obsm['spatial']
    boundary = try_alpha_shape(coords[v == region, :], alpha_max=alpha_max, alpha_min=alpha_min)
    return boundary


def alpha_shape(coords, alpha):
    """
    https://gist.github.com/dwyerk/10561690#gistcomment-2819818

    Compute the alpha shape (concave hull) of a set
    of points.
    :param coords: Iterable container of points.
    :param alpha: alpha value to influence the
        gooeyness of the border. Smaller numbers
        don't fall inward as much as larger numbers.
        Too large, and you lose everything!
    """
    if len(coords) < 4:
        raise ValueError

    tri = Delaunay(coords)
    triangles = coords[tri.vertices]
    edge_points = alpha_shape_sub(triangles, alpha)
    m = geometry.MultiLineString(edge_points.tolist())
    triangles = list(polygonize(m))
    if len(triangles) == 0:
        raise ValueError
    res = np.array(unary_union(triangles).boundary.coords)
    if len(res) == 4:
        raise ValueError
    return res


#@jit
def alpha_shape_sub(triangles, alpha):
    a = ((triangles[:, 0, 0] - triangles[:, 1, 0]) ** 2 + (
            triangles[:, 0, 1] - triangles[:, 1, 1]) ** 2) ** 0.5
    b = ((triangles[:, 1, 0] - triangles[:, 2, 0]) ** 2 + (
            triangles[:, 1, 1] - triangles[:, 2, 1]) ** 2) ** 0.5
    c = ((triangles[:, 2, 0] - triangles[:, 0, 0]) ** 2 + (
            triangles[:, 2, 1] - triangles[:, 0, 1]) ** 2) ** 0.5
    s = (a + b + c) / 2.0
    areas = (s * (s - a) * (s - b) * (s - c)) ** 0.5
    circums = a * b * c / (4.0 * areas)
    filtered = triangles[circums < (1.0 / alpha)]
    edge1 = filtered[:, (0, 1)]
    edge2 = filtered[:, (1, 2)]
    edge3 = filtered[:, (2, 0)]
    edge_points = np.unique(np.concatenate((edge1, edge2, edge3)), axis=0)
    return edge_points


def try_alpha_shape(coords, alpha_max, alpha_min):
    """
    Try to construct a shell around points for various alpha values, starting at alpha_max and
    decreasing by halves to alpha_min each time the construction fails.
    """
    if coords.shape[0] < 4:
        raise ValueError("coords must contain at least 4 points.")
    max_iters = 100
    alpha = alpha_max
    for _ in range(max_iters):
        try:
            boundary = alpha_shape(coords, alpha)
            return boundary
        except (NotImplementedError, ValueError):
            # This error is raised if the resulting geometry is not simply connected
            alpha = alpha * 0.5
            #print(alpha)
            if alpha < alpha_min:
                return None


def hotspot_closure(adata, edge_radius=0.001, processes=4, alpha_max=10,
                    verbose=False):
    with Pool(processes) as p:
        keys = [v for v in adata.obs if "hotspots_" in v]
        points = adata.obsm['spatial']

        boundaries = p.imap(hotspot_closure_sub, hotspot_closure_generator(adata, keys),
                            chunksize=1)
        if verbose:
            boundaries = tqdm(boundaries,
                              total=len(list(hotspot_closure_generator(adata, keys))))

        for key, c, boundary in boundaries:
            if boundary is None:
                continue

            path = mpl.path.Path(boundary)
            adata.obs[key][path.contains_points(points, radius=-edge_radius)] = c

    """
    keys = [v for v in adata.obs if "hotspots_" in v]
    points = adata.obsm['spatial']
    if verbose:
        keys = tqdm(keys)
    for key in keys:
        for c in adata.obs[key].cat.categories:
            try:
                boundary = compute_hotspot_boundary(adata, key, c, alpha_max=alpha_max)
                path = mpl.path.Path(boundary)
                adata.obs[key][path.contains_points(points, radius=-edge_radius)] = c
            except ValueError:
                continue
    """

def hotspot_closure_generator(adata, keys):
    for key in keys:
        for c in adata.obs[key].cat.categories:
            v = adata.obs[key]

            coords = adata.obsm['spatial']
            subcoords = coords[v == c, :].copy()
            yield (key, c, subcoords)

def hotspot_closure_sub(args):
    key, c, subcoords = args
    #return (key, c, None)
    try:
        return (key, c, try_alpha_shape(subcoords, alpha_max=10,
                               alpha_min=0.001))
    except ValueError:
        return (key, c, None)


def clear_hotspots(adata):
    adata.obs.drop(adata.obs.filter(regex='hotspots').columns.tolist(), axis=1, inplace=True)


def get_min_samples_from_density(adata, eps, density):
    """
    Helper function for choosing the min_samples parameter used in computing single-gene hotspots. Given adata object and the
    neighbor_eps parameter (how far to look), choose a min_samples corresponding to, on average, at least a fraction `density` of
    spots within neighbor_eps being active.
    """
    # Compute mean number of other spots within neighbor_eps
    coords = adata.obsm['spatial']
    tree = BallTree(coords)
    n_cells = len(coords)

    nearby_col = tree.query_radius(coords, eps)
    mean_neighbors = np.mean([len(v) for v in nearby_col]) - 1
    min_samples  = np.ceil(density * mean_neighbors).astype(np.int_)

    return min_samples
