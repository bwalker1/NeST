import scanpy as sc


class CellChat:
    def __init__(self):
        # set up to run the R code
        try:
            from rpy2 import robjects
            from rpy2.robjects.packages import importr
            import anndata2ri
        except ImportError:
            raise ImportError("Optional dependencies rpy2 and anndata2ri must be installed to"
                              " run the Cellchat wrapper.")

        anndata2ri.activate()
        self.robjects = robjects
        self.anndata2ri = anndata2ri

        # TODO: figure out what error happens if cellchat is not installed correctly
        self.cellchat = importr("CellChat")

    def run(self, adata, group_by="class"):
        adata = adata.copy()
        # converter does not currently seem to support sparse matrix entries
        try:
            adata.X = adata.X.todense()
        except AttributeError:
            pass

        # adata.X = np.exp(adata.X) + 1

        # discard extraneous information (which may or may not convert into R)
        adata.obs = sc.get.obs_df(adata, keys=["class"])

        self.robjects.r('''
                    f <- function(s, group.by) {
                        assay(s, "logcounts") = assay(s, "X")
                        #print(s)
                        #print(dim(pos))
                        # Save this for testing analysis
                        save(s, file="/Users/blw/Documents/data/sce.dat")

                        cellchat <- createCellChat(object = s, group.by = group.by)

                        print(cellchat)

                        CellChatDB <- CellChatDB.mouse
                        library(dplyr)
                        dplyr::glimpse(CellChatDB$interaction)

                        CellChatDB.use <- subsetDB(CellChatDB, search = "Secreted Signaling")
                        cellchat@DB = CellChatDB.use

                        cellchat <- subsetData(cellchat)
                        cellchat <- identifyOverExpressedGenes(cellchat)
                        cellchat <- identifyOverExpressedInteractions(cellchat)
                        cellchat <- projectData(cellchat, PPI.mouse)

                        #prior.thresholds <- list("Secreted Signaling"=0.1)
                        #pos <- matrix(rnorm(13938*2), 13938)
                        # TODO: figure out the right way to get the stuff loaded
                        library(Matrix)
                        cellchat <- computeCommunProb(cellchat, #prior.thresholds=prior.thresholds,
                                                      type="truncatedMean", trim=0.05)

                        print(cellchat)

                        df.net = subsetCommunication(cellchat)

                        df.net
                    }
                ''')

        out = self.robjects.r['f'](adata, group_by)

        return out
