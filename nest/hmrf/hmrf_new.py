from collections import deque
import scipy.sparse
import scipy.spatial
import scipy.stats
from tqdm import tqdm

from nest import get_neighbor_adjacency

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def hmrf(adata, regions, eps, max_iterations=200, verbose=False, dim=16, pca=True, beta=1):
    ncells = len(adata)

    # perform PCA on genes
    X = np.array(adata.X.toarray())
    if pca:
        pca_transform = PCA(n_components=dim)
        pca_transform.fit(X)
        pca = pca_transform.transform(X)
    else:
        if adata.shape[1] != dim:
            raise ValueError
        pca = X

    pca_t = np.transpose(pca)





    gamma = initialize_gamma()

    last_10_ll = deque()

    flag = False  # convergence stopping

    if max_iterations is not None:
        iterations = max_iterations
    else:
        iterations = max_iterations

    if verbose:
        pbar = tqdm(range(iterations), leave=False)
    else:
        pbar = range(iterations)

    for it in pbar:
        ll = np.zeros(shape=(ncells, regions))

        # Compute log-likelihood summed across all models
        for i, model in enumerate(likelihood_models):
            model.update_parameters(gamma)

            if model_weights is not None:
                weight = model_weights[i]
            else:
                weight = 1

            ll += weight * model.compute_log_likelihood(gamma)

        # Also do class size-based likelihood
        class_prob = np.mean(gamma, axis=0)
        ll += np.log(class_prob).reshape(1, regions)

        # Compute new class probabilities
        ll = ll
        gamma = scipy.special.softmax(ll, axis=1)

        model_ll = np.mean(ll * gamma)

        if tol is not None and it >= 10 and model_ll - max(last_10_ll) < tol:
            flag = True

        if len(last_10_ll) >= 10:
            last_10_ll.popleft()
        last_10_ll.append(model_ll)

        # compute overall model likelihood to watch for convergence
        class_ll = ll[:, class_labels].mean()

        if verbose:
            pbar.set_description(f"LL: {model_ll}")

        if flag:
            converged = True
            if verbose:
                pbar.set_description("Converged: LL %f" % model_ll)
            break

    if update_labels:
        adata.obs[label_name] = pd.Categorical(class_labels)



def initialize_gamma(adata, regions):
    try:
        data = adata.X.toarray()
    except TypeError:
        data = adata.X

    # do dimensionality reduction with PCA
    reduced_dims = 16
    pca = PCA(n_components=reduced_dims)
    data = pca.fit_transform(data)

    class_labels = KMeans(n_clusters=regions).fit(data).labels_
    labels_onehot = pd.get_dummies(class_labels, columns=range(regions)).to_numpy()

    # how much uncertainty to introduce in initial clusters
    eps = 0.1

    gamma = (1 - eps) * labels_onehot + eps / regions * np.ones(
        shape=labels_onehot.shape)

    return gamma


def update_parameters(gamma, pca):
    regions = gamma.shape[1]
    dim = pca.shape[1]
    gamma_sum = np.sum(gamma, axis=0)

    gamma_t = np.transpose(gamma)
    means = (gamma_t @ pca) / gamma_sum.reshape(regions, 1)

    pca_minus_mean = pca.reshape((-1, 1, dim)) - means.reshape(
        (1, regions, dim))
    A = pca_minus_mean.reshape(-1, regions, dim, 1) * \
        pca_minus_mean.reshape(-1, regions, 1, dim)

    covs = np.einsum('ij,jimn->imn', gamma_t, A) / gamma_sum.reshape((regions, 1, 1))

    return means, covs