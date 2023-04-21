import numpy as np
import pandas as pd
import scanpy as sc
from tqdm import tqdm
import leidenalg as la
import igraph as ig
try:
    from scipy.sparse import coo_array
except ImportError:
    # google colab compatibility
    from scipy.sparse import coo_matrix as coo_array
from scipy.spatial import KDTree


def segmentation_3d(adata, z_key="Bregma", output_key="segmentation_3d", inplace=True):
    """
    :param adata: Anndata object
    :param z_key: Column name in adata.obs containing the z-value associated with each observation.
    :param output_key: Column name in adata.obs to store segmentation membership in
    :param inplace: If `True`, modify adata. If `False` return a copy.

    :return: adata object with segmentation added to adata.obs, only if `inplace=False`.
    """
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
    n_pca = 8
    sc.tl.pca(adata, n_comps=n_pca)

    z_values = np.unique(adata.obs[z_key])

    for layer in tqdm(range(12)):
        adata_sub_1 = adata[adata.obs['Bregma'] == z_values[layer], :]
        X_pca = adata_sub_1.obsm['X_pca']
        xs = adata_sub_1.obsm['spatial']
        kd_1 = KDTree(xs)
        dist_1, nn_1 = kd_1.query(x=xs, k=range(2, kv + 2))
        dist_weights = np.exp(-(dist_1 / 0.04))

        kd_pca = KDTree(X_pca)
        dist_pca, nn_pca = kd_pca.query(x=X_pca, k=range(2, kv + 2))
        exp_weights = np.exp(-np.linalg.norm(X_pca.reshape((-1, 1, n_pca)) - X_pca[nn_1, :], axis=-1) / 2)

        # compute within_layer adjacency
        i = np.ravel([(i,) * kv for i in range(len(xs))])
        j = np.ravel(nn_1)
        data = np.ravel(dist_weights)
        within_adj_dist = coo_array((data, (i + layer_offset, j + layer_offset))).tocsr()

        i = np.ravel([(i,) * kv for i in range(len(xs))])
        j = np.ravel(nn_pca)
        data = np.ravel(exp_weights)
        within_adj_expr = coo_array((data, (i + layer_offset, j + layer_offset))).tocsr()

        within_adj = (alpha * within_adj_dist + (1 - alpha) * within_adj_expr).tocoo()

        within_adj_list.append(within_adj)

        # add to between layer adjacency
        if layer < 11:
            adata_sub_2 = adata[adata.obs['Bregma'] == z_values[layer + 1], :]
            xt = adata_sub_2.obsm['spatial']
            kd_2 = KDTree(xt)

            dist_12, nn_12 = kd_1.query(x=xt, k=k_inter)
            dist_21, nn_21 = kd_2.query(x=xs, k=k_inter)

            # C = cdist(adata_sub_1.obsm['spatial'], adata_sub_2.obsm['spatial'], metric='sqeuclidean')
            A = np.zeros((len(xs), len(xt)))

            for i, row in enumerate(nn_21):
                for j in row:
                    weight = 1 / k_inter
                    inter_layer_edges.append((i + layer_offset, j + layer_offset + len(xs), omega * weight))
                    A[i, j] = 1
            for j, row in enumerate(nn_12):
                for i in row:
                    weight = 1 / k_inter
                    inter_layer_edges.append((i + layer_offset, j + layer_offset + len(xs), omega * weight))
                    A[i, j] = 1

        layer_offset += len(xs)
        layer_offset_list.append(layer_offset)

    partitions = []
    for adj in within_adj_list:
        adj.resize((layer_offset, layer_offset))
        g = ig.Graph.Weighted_Adjacency(adj)
        g.vs['node_size'] = 1
        partitions.append(la.RBConfigurationVertexPartition(g, weights='weight', resolution_parameter=gamma))
        # partitions.append(la.CPMVertexPartition(g, weights='weight', resolution_parameter=gamma,
        #                                       node_sizes='node_size'))

    inter_layer_edges_arr = np.array(inter_layer_edges).T
    data = inter_layer_edges_arr[2, :]
    i = inter_layer_edges_arr[0, :].astype(np.int_)
    j = inter_layer_edges_arr[1, :].astype(np.int_)
    inter_layer_adj = coo_array((data, (i, j)))
    inter_layer_adj.resize(layer_offset, layer_offset)
    inter_g = ig.Graph.Weighted_Adjacency(inter_layer_adj)
    inter_g.vs['node_size'] = 0
    partitions.append(la.CPMVertexPartition(inter_g, weights='weight', resolution_parameter=0.0,
                                            node_sizes='node_size'))

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
