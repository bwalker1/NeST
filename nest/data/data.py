import os

import anndata
import pandas as pd
import scanpy as sc
import squidpy as sq
import numpy as np

from nest.utility import spatial_smoothing

SQUIDPY_DATASETS = ["seqfish", "slideseq", "merfish"]

VISIUM_DATASETS = [
    "V1_Breast_Cancer_Block_A_Section_1",
    "V1_Mouse_Brain_Sagittal_Anterior",
    "V1_Mouse_Brain_Sagittal_Posterior",
    "V1_Adult_Mouse_Brain_Coronal_Section_1", "V1_Breast_Cancer_Block_A_Section_2",
    "V1_Human_Heart", "V1_Human_Lymph_Node", "V1_Mouse_Brain_Sagittal_Posterior",
    "V1_Mouse_Brain_Sagittal_Posterior_Section_2", "V1_Mouse_Brain_Sagittal_Anterior",
    "V1_Mouse_Brain_Sagittal_Anterior_Section_2",
    "Targeted_Visium_Human_Cerebellum_Neuroscience", "Parent_Visium_Human_Cerebellum",
    "Targeted_Visium_Human_BreastCancer_Immunology",
    "Targeted_Visium_Human_OvarianCancer_Pan_Cancer",
    "Targeted_Visium_Human_OvarianCancer_Immunology"
    "V1_Human_Brain_Section_2",
    "V1_Adult_Mouse_Brain_Coronal_Section_1", "V1_Adult_Mouse_Brain_Coronal_Section_2",
    "Targeted_Visium_Human_SpinalCord_Neuroscience", "Parent_Visium_Human_SpinalCord",
    "Targeted_Visium_Human_Glioblastoma_Pan_Cancer", "Parent_Visium_Human_Glioblastoma",
    "Parent_Visium_Human_BreastCancer",
    "Parent_Visium_Human_OvarianCancer", "Targeted_Visium_Human_ColorectalCancer_GeneSignature",
    "Parent_Visium_Human_ColorectalCancer", "V1_Mouse_Kidney"]

SPATIAL_LIBD_DATASETS = ["Spatial_LIBD_%s" % item for item in
                         ["151507", "151671", "151673", "151509", "151510", "151669", "151670",
                          "151674", "151675",
                          "151672", "151676", "151508"]]


def get_squidpy_data(dataset, **kwargs):
    if dataset == "seqfish":
        adata = sq.datasets.seqfish()
        adata.uns['um_scale'] = 0.001
        adata.obs['class'] = adata.obs['celltype_mapped_refined']
    elif dataset == "merfish":
        adata = sq.datasets.merfish()
        adata.uns['um_scale'] = 0.001
        try:
            bregma = kwargs["bregma"]
        except KeyError:
            bregma = 1
        adata = adata[adata.obs.Bregma == bregma].copy()
        adata.obs['class'] = adata.obs['Cell_class']
        adata.raw = adata
        sc.pp.filter_cells(adata, min_counts=20)
    elif dataset == "slideseq":
        adata = sq.datasets.slideseqv2().raw.to_adata()
        adata.raw = adata.copy()
        adata.uns['um_scale'] = 1
        adata.obs["class"] = adata.obs["cluster"]

        # Rotate data so hippocampus is lengthwise "horizontal"
        X = adata.obsm['spatial']
        # Center on origin
        X -= np.mean(X, axis=0)
        theta = np.radians(20)
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]).T
        X = X @ R
        adata.obsm['spatial'] = X
    else:
        return ValueError("Dataset not recognized")

    return adata


def load_stereo_seq_data(dataset_dir):
    file = os.path.join(dataset_dir, "stereoseq", "adata.h5ad")
    adata = anndata.read_h5ad(file)
    return adata


def get_data(dataset, dataset_dir=None, normalize=True, **kwargs):
    if dataset in SQUIDPY_DATASETS:
        adata = get_squidpy_data(dataset=dataset, **kwargs)
    elif dataset == "stereoseq":
        adata = load_stereo_seq_data(dataset_dir)
    elif dataset in SPATIAL_LIBD_DATASETS:
        if dataset_dir is None:
            raise ValueError("For spatialLIBD data, must set dataset_dir to directory containing the data.")
        expr_dir = os.path.join(dataset_dir, dataset)
        h5ad_path = os.path.join(expr_dir, "adata.h5ad")
        try:
            adata = anndata.read_h5ad(h5ad_path)
        except FileNotFoundError:
            adata = sc.read_10x_mtx(expr_dir).copy()
            coord_fp = os.path.join(expr_dir, "spatial_coords.csv")
            coord_df = pd.read_csv(coord_fp).values.astype(float)

            adata.obsm['spatial'] = coord_df[:, [1, 0]]

            info_df = pd.read_csv(os.path.join(expr_dir, "spot_info.csv"))
            clusters = info_df["layer_guess_reordered"].values.astype(str)
            adata.obs['class'] = pd.Categorical(clusters)
            convert_ensembl_names(adata)

            # Cache this adata so we don't have to redo all this next time
            adata.write(h5ad_path)
    elif dataset in VISIUM_DATASETS:
        adata = sc.datasets.visium_sge(dataset, include_hires_tiff=False)
        adata.var_names_make_unique()
    else:
        raise ValueError("Unrecognized dataset")

    sc.pp.filter_genes(adata, min_counts=1)  # only consider genes with more than 1 count
    if normalize:
        sc.pp.normalize_total(adata)
    if 'log1p' not in adata.uns:
        sc.pp.log1p(adata)

    return adata


def convert_ensembl_names(adata):
    try:
        import mygene
    except ImportError:
        raise RuntimeError("Package mygene must be installed to run convert_ensembl_names")

    mg = mygene.MyGeneInfo()
    ens = adata.var_names
    res = mg.querymany(ens, scope="ensembl.gene", returnall=True)

    name_map = {}
    for item in res['out']:
        try:
            name_map[item['query']] = item['symbol']
        except KeyError:
            # No symbol was returned
            name_map[item['query']] = item['query']

    for gene in res['missing']:
        name_map[gene] = gene

    adata.var_names = pd.Index([name_map[gene] for gene in adata.var_names])


def default_parameters(dataset):
    if "V1_Breast_Cancer_Block_A" in dataset:
        neighbor_eps = 300
        min_samples = 4
        hotspot_min_size = 15
    elif dataset in VISIUM_DATASETS or dataset == "visium":
        neighbor_eps = 170
        min_samples = 4
        hotspot_min_size = 15
    elif dataset == "seqfish":
        neighbor_eps = 0.12
        min_samples = 10
        hotspot_min_size = 25
    elif dataset == "merfish":
        neighbor_eps = 0.06
        min_samples = 5
        hotspot_min_size = 5
    elif dataset == "slideseq":
        neighbor_eps = 75
        min_samples = 5
        hotspot_min_size = 50
    else:
        raise ValueError

    return neighbor_eps, min_samples, hotspot_min_size
