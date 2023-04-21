import warnings
import scipy.sparse

import numpy as np

from sklearn.preprocessing import normalize
from scipy.spatial import Delaunay

from nest.utility import get_neighbor_adjacency



def compute_spatial_transport_matrices(adata,
                                       secreted_std,
                                       secreted_threshold_cutoff,
                                       contact_threshold=None,
                                       z_key=None):
    """
    Return the spatial prior belief of possibility of interaction between every possible pair of
    cells for both secreted signaling and cell-cell contact type interactions
    """

    ncells = adata.shape[0]

    # gaussian diffusion kernel - threshold is two standard deviations
    sigma = secreted_std

    def weight_fn(x):
        return np.exp(-0.5 * ((x / (2 * sigma)) ** 2))

    coords = adata.obsm['spatial']
    if z_key is not None:
        z = np.array(adata.obs[z_key]).reshape(-1, 1)
        coords = np.concatenate([coords, z], axis=1)
    transport_secreted = get_neighbor_adjacency(coords, eps=sigma * secreted_threshold_cutoff,
                                                weight=weight_fn)
    if z_key is None:
        # delauney for direct contact
        tri = Delaunay(adata.obsm['spatial'])
        indptr, indices = tri.vertex_neighbor_vertices
        data = np.ones(shape=indices.shape)
        transport_contact = scipy.sparse.csr_matrix((data, indices, indptr), shape=[ncells, ncells])

        # filter out edges in delauney triangulation that are too long
        if contact_threshold is not None:
            transport_contact = transport_contact.multiply(
                get_neighbor_adjacency(coords, eps=contact_threshold))

        # normalize across number of neighbors
        transport_contact = normalize(transport_contact, norm='l1', axis=1)
        adata.obsp['transport_contact'] = transport_contact

    # transport_secreted = normalize(transport_secreted, norm='l1', axis=1)


    adata.obsp['transport_secreted'] = transport_secreted
