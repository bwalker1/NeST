from collections import deque

import numpy as np
import pandas as pd
import scipy.sparse
import scipy.spatial
import scipy.stats
import sklearn
from tqdm import tqdm

from nest import get_neighbor_adjacency


class GaussianExpressionModel:
    def __init__(self, dim=16, pca=True):
        self.dim = dim
        self.use_pca = pca

        # initialized later
        self.means = None
        self.covs = None

        self.regions = None
        self.adata = None

    def set_data(self, adata, regions):
        self.regions = regions
        self.adata = adata

        # perform PCA on genes
        try:
            X = np.array(self.adata.X.toarray())
        except AttributeError:
            X = self.adata.X
        if self.use_pca:
            self.pca_transform = sklearn.decomposition.PCA(n_components=self.dim)
            self.pca_transform.fit(X)
            self.pca = self.pca_transform.transform(X)
        else:
            if adata.shape[1] != self.dim:
                raise ValueError
            self.pca = X

        self.pca_t = np.transpose(self.pca)

    def update_data(self, adata):
        self.adata = adata
        # apply the PCA transformation, but without re-fitting
        X = self.adata.X.log1p().toarray()
        self.pca = self.pca_transform.transform(X)
        self.pca_t = np.transpose(self.pca)

    def update_parameters(self, gamma):
        gamma_sum = np.sum(gamma, axis=0)

        gamma_t = np.transpose(gamma)
        self.means = (gamma_t @ self.pca) / gamma_sum.reshape(self.regions, 1)

        pca_minus_mean = self.pca.reshape((-1, 1, self.dim)) - self.means.reshape(
            (1, self.regions, self.dim))
        A = pca_minus_mean.reshape(-1, self.regions, self.dim, 1) * \
            pca_minus_mean.reshape(-1, self.regions, 1, self.dim)

        self.covs = np.einsum('ij,jimn->imn', gamma_t, A) / gamma_sum.reshape((self.regions, 1, 1))

    def compute_log_likelihood(self, gamma):
        # does not use gamma
        ll = np.zeros(shape=gamma.shape)
        for i in range(self.regions):
            ll[:, i] = scipy.stats.multivariate_normal.logpdf(x=self.pca, mean=self.means[i, :],
                                                              cov=self.covs[i, :, :],
                                                              allow_singular=True)
        return ll


class HMRF:
    """
    Likelihood model for HMRF
    """

    def __init__(self, beta, threshold=None, k=None, cap=None):
        self.threshold = threshold
        self.k = k
        self.beta = beta
        self.cap = cap

        # To be set later
        self.adjacency = None
        self.regions = None

    def set_data(self, adata, regions):
        self.regions = regions
        coords = adata.obsm['spatial']

        # This function automatically switches depending on which of threshold/k is not None
        self.adjacency = get_neighbor_adjacency(coords, eps=self.threshold, k=self.k)

    def compute_log_likelihood(self, gamma):
        log_likelihood = (self.adjacency @ gamma)
        if self.cap is not None:
            log_likelihood[log_likelihood > self.cap] = self.cap

        log_likelihood = self.beta * log_likelihood
        return log_likelihood

    def update_parameters(self, gamma):
        pass


class HMRFSegmentationModel:
    def __init__(self, adata, regions, eps=None, k=None, model_weights=None, label_name=None,
                 max_iterations=200, verbose=False, dim=16, pca=True, beta=1):
        self.adata = adata
        self.likelihood_models = [GaussianExpressionModel(dim=dim, pca=pca),
                                  HMRF(beta=beta, threshold=eps, k=k)]
        # number of regions to segment into
        self.regions = regions
        self.model_weights = model_weights
        self.label_name = label_name
        self.max_iterations = max_iterations
        self.verbose = verbose

        # Set in fit function

        self.gamma = None
        self.converged = False

        self.model_ll = None
        self.class_ll = None

        for model in self.likelihood_models:
            model.set_data(adata, self.regions)

    def fit(self, max_iterations=None, update_labels=False, tol=1e-6, metric=None, verbose=False,
            **kwargs):
        ncells = len(self.adata)

        # if no current class information, run initialization
        if self.gamma is None:
            self.initialize_gamma()

        last_10_ll = deque()

        flag = False  # convergence stopping

        if max_iterations is not None:
            iterations = max_iterations
        else:
            iterations = self.max_iterations

        if verbose:
            pbar = tqdm(range(iterations), leave=False)
        else:
            pbar = range(iterations)

        for it in pbar:
            ll = np.zeros(shape=(ncells, self.regions))

            # Compute log-likelihood summed across all models
            for i, model in enumerate(self.likelihood_models):
                model.update_parameters(self.gamma)

                if self.model_weights is not None:
                    weight = self.model_weights[i]
                else:
                    weight = 1

                ll += weight * model.compute_log_likelihood(self.gamma)

            # Also do class size-based likelihood
            class_prob = np.mean(self.gamma, axis=0)
            ll += np.log(class_prob).reshape(1, self.regions)

            # Compute new class probabilities
            self.ll = ll
            self.gamma = scipy.special.softmax(ll, axis=1)

            self.model_ll = np.mean(ll * self.gamma)

            if tol is not None and it >= 10 and self.model_ll - max(last_10_ll) < tol:
                flag = True

            if len(last_10_ll) >= 10:
                last_10_ll.popleft()
            last_10_ll.append(self.model_ll)

            # compute overall model likelihood to watch for convergence
            self.class_ll = ll[:, self.class_labels].mean()

            if self.verbose:
                pbar.set_description(f"LL: {self.model_ll}")

            if flag:
                self.converged = True
                if self.verbose:
                    pbar.set_description("Converged: LL %f" % self.model_ll)
                break

        if update_labels:
            self.adata.obs[self.label_name] = pd.Categorical(self.class_labels)

    def initialize_gamma(self):
        try:
            data_raw = self.adata.X.toarray()
        except (TypeError, AttributeError) as e:
            data_raw = self.adata.X

        self.kmeans_initialization(data_raw)

    def kmeans_initialization(self, data):
        # do dimensionality reduction with PCA
        reduced_dims = 16
        pca = sklearn.decomposition.PCA(n_components=reduced_dims)
        data = pca.fit_transform(data)

        class_labels = sklearn.cluster.KMeans(n_clusters=self.regions).fit(data).labels_
        labels_onehot = pd.get_dummies(class_labels, columns=range(self.regions)).to_numpy()

        # how much uncertainty to introduce in initial clusters
        eps = 0.1

        self.gamma = (1 - eps) * labels_onehot + eps / self.regions * np.ones(
            shape=labels_onehot.shape)

    @property
    def class_labels(self):
        labels = np.argmax(self.gamma, axis=1)
        return labels
