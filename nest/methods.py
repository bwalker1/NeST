"""
Code for running other spatial segmentation methods
"""
import contextlib

import numpy as np
import pandas as pd
import sys


# TODO: do all the citations and stuff properly

class CellChat:
    def __init__(self):
        pass

    def run(self, adata, group_by="class"):
        # set up to run the R code
        import rpy2.robjects as robjects
        from rpy2.robjects.packages import importr
        import anndata2ri
        anndata2ri.activate()
        #from rpy2.robjects import pandas2ri
        #pandas2ri.activate()
        self.robjects = robjects
        self.anndata2ri = anndata2ri

        self.cellchat = importr("CellChat")

        # converter does not currently seem to support sparse matrix entries
        adata = adata.copy()
        try:
            adata.X = adata.X.todense()
        except AttributeError:
            pass

        # get rid of extraneous columns that may not convert properly anyway
        adata.obs = adata.obs.loc[:, [group_by]]

        self.robjects.r('''
                    f <- function(s, pos=NULL) {
                        assay(s, "logcounts") = assay(s, "X")

                        cellchat <- createCellChat(object = s, group.by = "%s")

                        print(cellchat)

                        CellChatDB <- CellChatDB.mouse

                        CellChatDB.use <- CellChatDB
                        cellchat@DB = CellChatDB.use

                        cellchat <- subsetData(cellchat)
                        cellchat <- identifyOverExpressedGenes(cellchat)
                        cellchat <- identifyOverExpressedInteractions(cellchat)
                        cellchat <- projectData(cellchat, PPI.mouse)

                        library(Matrix)
                        cellchat <- computeCommunProb(cellchat, type="truncatedMean", trim=0.05)

                        print(cellchat)

                        df.net = subsetCommunication(cellchat)

                        df.net
                    }
                ''' % group_by)

        out = self.robjects.r['f'](adata)
        df = pd.DataFrame(out).T
        df.columns = out.colnames
        return df

    def cellchat_score(self, adata, interaction, key="cellchat_score", group_by="class", rerun=False):
        try:
            if rerun:
                raise KeyError
            res = adata.uns['cellchat_res']
        except KeyError as e:
            res = self.run(adata, group_by)
            adata.uns['cellchat_res'] = res
        v = res[res['interaction_name_2'] == interaction][np.array(['target', 'prob'])].groupby('target').sum()
        # TODO: check that we have a nonzero number of rows in v
        v.index = adata.obs[group_by].cat.categories[v.index - 1]
        # fill in any missing categories with 0 for next step lookup to work
        for s in set(np.unique(adata.obs[group_by])) - set(v.index):
            v.loc[s, 'prob'] = 0
        adata.obs[key] = np.array(v.loc[adata.obs[group_by], 'prob'])

class BayesSpace:
    """
    Bayesspace method from https://github.com/edward130603/BayesSpace
    """

    def __init__(self, regions, label_name="class_bayesspace"):
        self.regions = regions
        self.label_name = label_name

    def fit(self, adata, verbose=False, **kwargs):
        # set up to run the R code
        import rpy2.robjects as robjects
        from rpy2.robjects.packages import importr
        import anndata2ri
        anndata2ri.activate()
        self.robjects = robjects
        self.anndata2ri = anndata2ri

        self.sce = importr("SingleCellExperiment")
        self.bayesspace = importr("BayesSpace")

        platform = "Visium"

        # converter does not currently seem to support sparse matrix entries
        adatasub = adata.copy()
        adatasub.obs = pd.DataFrame({'row': adatasub.obs['array_row'],
                                    'col': adatasub.obs['array_col']},
                                    index=adatasub.obs.index)
        try:
            adatasub.X = adatasub.X.toarray()
        except AttributeError:
            pass

        adatasub.raw = None
        adatasub.obsp = None
        try:
            del adatasub.layers['exp']
        except KeyError:
            pass

        self.robjects.r('''
            f <- function(s, q) {
                assay(s, "logcounts") = assay(s, "X")
                s <- spatialPreprocess(s, platform="Visium", n.PCs=16, n.HVGs=2000, log.normalize=FALSE)
                s <- spatialCluster(s, q=q, platform="Visium", d=15, init.method="mclust", model="t", gamma=2, nrep=200, burn.in=100, save.chain=FALSE)
                print("Completed clustering")
                print(s)
                rowData(s)$is.HVG <- NULL
                metadata(s)$BayesSpace.data <- NULL
                s
            }
        ''')
        output = sys.stdout if verbose else None
        utils = importr('utils')
        out = self.robjects.r['f'](adatasub, self.regions)

        # copy the output labels back over
        adata.obs[self.label_name] = out.obs['spatial.cluster']

        return




class SpaGCN:
    def __init__(self, regions, label_name="class_spagcn"):
        self.regions = regions
        self.label_name = label_name

    def fit(self, adata, img=None, verbose=False, **kwargs):
        import SpaGCN as spg
        import scanpy as sc
        import torch
        import random
        import sys
        import contextlib

        output = sys.stdout if verbose else None
        adatasub = adata.copy()
        with contextlib.redirect_stdout(output):
            adatasub.var_names_make_unique()
            try:
                adatasub.X = adatasub.X.toarray()
            except AttributeError:
                pass
            adatasub.X = np.exp(adatasub.X) - 1
            spg.prefilter_genes(adatasub, min_cells=3)  # avoiding all genes are zeros
            spg.prefilter_specialgenes(adatasub)
            # Normalize and take log for UMI
            sc.pp.normalize_per_cell(adatasub)
            sc.pp.log1p(adatasub)

            # get adjacency matrix
            # Calculate adjacent matrix
            if img is not None:
                s = 1
                b = 49
                adj = spg.calculate_adj_matrix(x=x_pixel, y=y_pixel, x_pixel=x_pixel,
                                               y_pixel=y_pixel, image=img,
                                               beta=b,
                                               alpha=s, histology=True)
            else:
                x = adatasub.obsm['spatial'][:, 0]
                y = adatasub.obsm['spatial'][:, 1]
                adj = spg.calculate_adj_matrix(x=x, y=y, histology=False)

            p = 0.5
            # Find the l value given p
            l = spg.search_l(p, adj, start=0.001, end=1000, tol=0.01, max_run=100)

            # Set seed
            if False:
                r_seed = t_seed = n_seed = 100
                # Set seed
                random.seed(r_seed)
                torch.manual_seed(t_seed)
                np.random.seed(n_seed)
            # Search for suitable resolution
            # res = spg.search_res(adata, adj, l, n_clusters, start=0.7, step=0.1, tol=5e-3, lr=0.05, max_epochs=20,
            #                     r_seed=r_seed, t_seed=t_seed, n_seed=n_seed)

            clf = spg.SpaGCN()
            clf.set_l(l)

            # Run
            clf.train(adatasub, adj, init_spa=True, init="kmeans", n_clusters=self.regions, tol=5e-3,
                      lr=0.05,
                      max_epochs=500)
            y_pred, prob = clf.predict()
            adata.obs[self.label_name] = pd.Categorical(y_pred)

    @property
    def class_labels(self):
        return self.y_pred



class STLearn:
    def __init__(self):
        import stlearn
        self.stlearn = stlearn

    def fit(self, adata, img=None, **kwargs):
        st = self.stlearn

        data = st.convert_scanpy(adata)

        st.pp.filter_genes(data, min_cells=1)
        st.pp.normalize_total(data)
        st.pp.log1p(data)

        # run PCA for gene expression data
        st.em.run_pca(data, n_comps=50)

        data_SME = data.copy()
        # apply stSME to normalise log transformed data
        # with weights from morphological Similarly and physcial distance
        st.spatial.SME.SME_normalize(data_SME, use_data="raw",
                                     weights="weights_matrix_pd_md")
        data_SME.X = data_SME.obsm['raw_SME_normalized']
        st.pp.scale(data_SME)
        st.em.run_pca(data_SME, n_comps=50)

        # K-means clustering on stSME normalised PCA
        st.tl.clustering.kmeans(data_SME, n_clusters=17, use_data="X_pca", key_added="X_pca_kmeans")

        # louvain clustering on stSME normalised data
        st.pp.neighbors(data_SME, n_neighbors=20, use_rep='X_pca')
        st.tl.clustering.louvain(data_SME)

        self.stdata = data_SME
        print(self.stdata)

    @property
    def class_labels(self):
        raise NotImplementedError