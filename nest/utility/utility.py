import scanpy as sc
import pandas as pd
import numpy as np
import anndata
import warnings
from tqdm import tqdm

from scipy.sparse import csr_matrix
from sklearn.neighbors import BallTree


def get_activity_dataframe(adata):
    obs_names = list(adata.obs)
    activity_names = [v for v in obs_names if "activity_" in v]
    return sc.get.obs_df(adata, keys=activity_names)


def write(adata, name):
    adata = adata.copy()
    adata.obs.to_pickle(f'{name}.pkl')
    adata.obs = pd.DataFrame(index=adata.obs.index)
    adata.write(f'{name}.h5ad')


def read(name):
    adata = anndata.read_h5ad(f'{name}.h5ad')
    adata.obs = pd.read_pickle(f'{name}.pkl')
    return adata


def spatial_smoothing(adata, mode=None, cutoff=30):
    smoothing_matrix = get_neighbor_adjacency(adata.obsm['spatial'], eps=cutoff)
    data = adata.X
    data2 = np.empty(data.shape)
    for idx in tqdm(range(smoothing_matrix.shape[0])):
        row = smoothing_matrix[idx, :]
        subdata = data[row.nonzero()[1], :].toarray()
        q1 = np.quantile(subdata, 0.2, axis=0)
        q2 = np.quantile(subdata, 0.8, axis=0)
        data2[idx, :] = np.mean([q1, q2], axis=0)
    adata.X = csr_matrix(data2)


def get_neighbor_adjacency(coords, eps=None, k=None, weight=None, return_row_col=False):
    """
    Compute either k-nearest neighbor or epsilon-radius neighbor adjacency graph
    :param coords: Spatial coordinates of each cell
    :param eps: radius cutoff for epsilon-radius neighbors
    :param k: number of nearest neighbors for k-nearest neighbors
    :param weight: Function mapping distance to value in weighted adjacency (default: f(x) = 1)
    :param return_row_col: If true, return the row and column variables from sparse coo format
    :return:
    """
    tree = BallTree(coords)
    n_cells = len(coords)
    if eps is not None and k is not None:
        warnings.warn('k is ignored if eps is provided')

    if eps is not None:
        nearby_col, dist = tree.query_radius(coords, eps, return_distance=True)
        coords = []
        nearby_row = []
        for i, col in enumerate(nearby_col):
            nearby_col[i] = col
            if weight is not None:
                coords.append(weight(dist[i]))
            nearby_row.append(i * np.ones(shape=col.shape, dtype=col.dtype))
    elif k is not None:
        nearby_col = tree.query(coords, k,
                                return_distance=False)  # [:, 1:]  # remove self connections
        nearby_row = []
        for i, col in enumerate(nearby_col):
            nearby_row.append(i * np.ones(shape=col.shape, dtype=col.dtype))
    else:
        raise ValueError("Provide either distance eps or number of neighbors k")

    nearby_row = np.concatenate(nearby_row)
    nearby_col = np.concatenate(nearby_col)
    if weight is None or k is not None:
        coords = np.ones(shape=nearby_row.shape)
    else:
        coords = np.concatenate(coords)

    nearby = csr_matrix((coords, (nearby_row, nearby_col)),
                        shape=(n_cells, n_cells))

    if return_row_col:
        return nearby, nearby_row, nearby_col

    return nearby
