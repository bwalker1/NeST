import warnings
import scipy.sparse
import scanpy as sc
from scipy.sparse import coo_array
import leidenalg as la
from scipy.spatial import KDTree
from tqdm import tqdm
import igraph as ig
import pandas as pd

import numpy as np

from sklearn.preprocessing import normalize
from scipy.spatial import Delaunay

from nest.utility import get_neighbor_adjacency


def compute_spatial_transport_matrices(
    adata, secreted_std, secreted_threshold_cutoff, contact_threshold=None, z_key=None
):
    """
    Return the spatial prior belief of possibility of interaction between every possible pair of
    cells for both secreted signaling and cell-cell contact type interactions
    """

    ncells = adata.shape[0]

    # gaussian diffusion kernel - threshold is two standard deviations
    sigma = secreted_std

    def weight_fn(x):
        return np.exp(-0.5 * ((x / (2 * sigma)) ** 2))

    coords = adata.obsm["spatial"]
    if z_key is not None:
        z = np.array(adata.obs[z_key]).reshape(-1, 1)
        coords = np.concatenate([coords, z], axis=1)
    transport_secreted = get_neighbor_adjacency(
        coords, eps=sigma * secreted_threshold_cutoff, weight=weight_fn
    )
    if z_key is None:
        # delauney for direct contact
        tri = Delaunay(adata.obsm["spatial"])
        indptr, indices = tri.vertex_neighbor_vertices
        data = np.ones(shape=indices.shape)
        transport_contact = scipy.sparse.csr_matrix(
            (data, indices, indptr), shape=[ncells, ncells]
        )

        # filter out edges in delauney triangulation that are too long
        if contact_threshold is not None:
            transport_contact = transport_contact.multiply(
                get_neighbor_adjacency(coords, eps=contact_threshold)
            )

        # normalize across number of neighbors
        transport_contact = normalize(transport_contact, norm="l1", axis=1)
        adata.obsp["transport_contact"] = transport_contact

    # transport_secreted = normalize(transport_secreted, norm='l1', axis=1)

    adata.obsp["transport_secreted"] = transport_secreted


def segmentation_3d(
    adata, z_key="Bregma", output_key="segmentation_3d", inplace=True, n_pca=8
):
    if not inplace:
        adata = adata.copy()
    kv = 20
    k_inter = 5
    gamma = 1
    omega = 0.0001
    alpha = 0.2

    layer_offset = 0
    layer_offset_list = [0]

    within_adj_list = []
    inter_layer_edges = []

    # PCA reduction over all layers
    sc.tl.pca(adata, n_comps=n_pca)

    z_values = np.unique(adata.obs[z_key])

    for layer in tqdm(range(12)):
        adata_sub_1 = adata[adata.obs["Bregma"] == z_values[layer], :]
        X_pca = adata_sub_1.obsm["X_pca"]
        xs = adata_sub_1.obsm["spatial"]
        kd_1 = KDTree(xs)
        dist_1, nn_1 = kd_1.query(x=xs, k=range(2, kv + 2))
        dist_weights = np.exp(-(dist_1 / 0.04))

        kd_pca = KDTree(X_pca)
        dist_pca, nn_pca = kd_pca.query(x=X_pca, k=range(2, kv + 2))
        exp_weights = np.exp(
            -np.linalg.norm(X_pca.reshape((-1, 1, n_pca)) - X_pca[nn_1, :], axis=-1) / 2
        )

        # compute within_layer adjacency
        i = np.ravel([(i,) * kv for i in range(len(xs))])
        j = np.ravel(nn_1)
        data = np.ravel(dist_weights)
        within_adj_dist = coo_array(
            (data, (i + layer_offset, j + layer_offset)), shape=(len(adata), len(adata))
        ).tocsr()

        i = np.ravel([(i,) * kv for i in range(len(xs))])
        j = np.ravel(nn_pca)
        data = np.ravel(exp_weights)
        within_adj_expr = coo_array(
            (data, (i + layer_offset, j + layer_offset)), shape=(len(adata), len(adata))
        ).tocsr()

        try:
            within_adj = (
                alpha * within_adj_dist + (1 - alpha) * within_adj_expr
            ).tocoo()
        except ValueError:
            print(within_adj_dist.shape, within_adj_expr.shape)

        within_adj_list.append(within_adj)

        # add to between layer adjacency
        if layer < 11:
            adata_sub_2 = adata[adata.obs["Bregma"] == z_values[layer + 1], :]
            xt = adata_sub_2.obsm["spatial"]
            kd_2 = KDTree(xt)

            dist_12, nn_12 = kd_1.query(x=xt, k=k_inter)
            dist_21, nn_21 = kd_2.query(x=xs, k=k_inter)

            # C = cdist(adata_sub_1.obsm['spatial'], adata_sub_2.obsm['spatial'], metric='sqeuclidean')
            A = np.zeros((len(xs), len(xt)))

            for i, row in enumerate(nn_21):
                for j in row:
                    weight = 1 / k_inter
                    inter_layer_edges.append(
                        (i + layer_offset, j + layer_offset + len(xs), omega * weight)
                    )
                    A[i, j] = 1
            for j, row in enumerate(nn_12):
                for i in row:
                    weight = 1 / k_inter
                    inter_layer_edges.append(
                        (i + layer_offset, j + layer_offset + len(xs), omega * weight)
                    )
                    A[i, j] = 1

        layer_offset += len(xs)
        layer_offset_list.append(layer_offset)

    partitions = []
    for adj in within_adj_list:
        adj.resize((layer_offset, layer_offset))
        g = ig.Graph.Weighted_Adjacency(adj)
        g.vs["node_size"] = 1
        partitions.append(
            la.RBConfigurationVertexPartition(
                g, weights="weight", resolution_parameter=gamma
            )
        )

    inter_layer_edges_arr = np.array(inter_layer_edges).T
    data = inter_layer_edges_arr[2, :]
    i = inter_layer_edges_arr[0, :].astype(np.int_)
    j = inter_layer_edges_arr[1, :].astype(np.int_)
    inter_layer_adj = coo_array((data, (i, j)))
    inter_layer_adj.resize((layer_offset, layer_offset))
    inter_g = ig.Graph.Weighted_Adjacency(inter_layer_adj)
    inter_g.vs["node_size"] = 0
    partitions.append(
        la.CPMVertexPartition(
            inter_g, weights="weight", resolution_parameter=0.0, node_sizes="node_size"
        )
    )

    optimiser = la.Optimiser()

    max_iters = 3
    for _ in tqdm(range(max_iters)):
        diff = optimiser.optimise_partition_multiplex(partitions)
        if diff == 0:
            break

    full_membership = np.array(partitions[0].membership)

    adata.obs[output_key] = pd.Categorical(full_membership)

    if not inplace:
        return adata
