import pandas as pd
import numpy as np

from tqdm import tqdm

from nest.cci.spatial import compute_spatial_transport_matrices
from nest.data import load_database

from scipy.stats import mannwhitneyu
from scipy.stats.mstats import gmean
try:
    from scipy.sparse import csc_array
except ImportError:
    # scipy 1.7 compatibility
    from scipy.sparse import csc_matrix as csc_array

import warnings


def compute_activity(adata, secreted_std, contact_threshold, permutations=100, K=0.5,
                     min_active_count=1,
                     sig_threshold=0.95, verbose=False, perform_permutation=False,
                     secreted_threshold_cutoff=2, save_activity=False,
                     interactions=None, z_key=None):
    """
    Compute the activity scores for all interactions across cells in adata. All output is stored in adata
    :param adata: anndata object
    :param permutations: number of permutation samples used in computing significance cutoffs (default 100)
    :param K: K parameter in Hill function
    :param min_active_count: minimum number of cells identified as active in order for interaction to be included in
    result (default 0)
    :param sig_threshold: quantile of permutation test bootstrap distribution used for significance cutoff
    :param secreted_std: Standard deviation of secreted Gaussian diffusion
    :param secreted_threshold_cutoff: Maximum number of standard deviations to compute (outside is thresholded to 0)
    :param contact_threshold: Maximum transport distance for contact interactions (must also be adjacent in Delauney
    triangulation)
    :param verbose: If `True`, prints out additional information while running (default: `False`)
    :param save_activity: If `True`, computed activity scores are saved in adata.obs
    :param interactions: list of interactions to compute activity for
    """

    # Check that filter_interactions has been run
    try:
        filtered_interactions = adata.uns["interactions"]
    except KeyError:
        interaction = load_database()
        filter_interactions(interaction, adata)
        filtered_interactions = adata.uns["interactions"]

    # Get a reduced view of adata containing only relevant genes
    adata_filtered = filter_adata(adata)
    compute_spatial_transport_matrices(adata,
                                       secreted_std=secreted_std,
                                       secreted_threshold_cutoff=secreted_threshold_cutoff,
                                       contact_threshold=contact_threshold,
                                       z_key=z_key)
    transport_secreted = adata.obsp['transport_secreted']
    try:
        transport_contact = adata.obsp['transport_contact']
    except KeyError:
        # TODO: implement contact + 3D
        transport_contact = None

    # Here we use the raw values
    try:
        expr = adata_filtered.X.toarray()
    except AttributeError:
        expr = adata_filtered.X
    # expr = np.log(1+expr)
    # Normalize each column to 0-1
    expr /= np.max(expr, axis=0)
    expr = csc_array(expr)
    # compute the L_i R_j score for each interaction and pair of genes
    activity_matrix_cols = {}
    pathways = {}
    interaction_to_pathway = {}

    if perform_permutation:
        permutation_vectors = []
        n_cells = len(adata_filtered)
        expr_boot = []
        # A_boot = []
        for i in range(permutations):
            if z_key is None:
                permutation = np.random.permutation(n_cells)
                permutation_vectors.append(permutation)
                expr_boot.append(expr[permutation, :])
            else:
                permutation = np.arange(n_cells)
                for val in np.unique(adata.obs[z_key]):
                    inds = np.where(adata.obs[z_key] == val)[0]
                    inds_permuted = np.random.permutation(inds)
                    permutation[inds] = inds_permuted
                expr_boot.append(expr[permutation, :])

    # track which interactions still remain after activity filtering
    cutoffs = {}
    rows = filtered_interactions.iterrows()
    if verbose:
        print("Computing activity scores")
        rows = tqdm(rows, total=filtered_interactions.shape[0])
    for k, row in rows:
        pathway = row["pathway_name"]
        interaction_name = row["interaction_name"]

        if interactions is not None and interaction_name not in interactions:
            pass

        if row["secreted"]:
            transport = transport_secreted
        else:
            transport = transport_contact

        if transport is None:
            continue

        if pathway not in pathways:
            pathways[pathway] = 1
        else:
            pathways[pathway] += 1

        i = np.where(adata_filtered.var_names == row["ligand"])[0]
        j = np.where(adata_filtered.var_names == row["receptor"])[0]

        ligand_i, receptor_j = expr[:, i], expr[:, j]
        activity = receptor_j.multiply(transport.dot(ligand_i))
        # Apply hill function
        if K is not None:
            activity.data = activity.data / (K + activity.data)
        activity = activity.toarray().reshape(-1)

        # filter out interactions with a low number of possibly interacting cells
        nonzero = np.count_nonzero(activity)
        if nonzero < min_active_count:
            continue

        activity_matrix_cols[interaction_name] = activity

        if perform_permutation:
            vals = np.array([])
            for kk in range(permutations):
                ligand_i_boot, Rj_boot = expr_boot[kk][:, i], expr_boot[kk][:, j]

                activity_boot = Rj_boot.multiply(transport.dot(ligand_i_boot))
                if K is not None:
                    activity_boot.data = activity_boot.data / (K + activity_boot.data)
                activity_boot = activity_boot.toarray().reshape(-1)
                vals = np.concatenate([vals, activity_boot])
            # TODO: check if we can avoid the zeros here to speed it up a little
            cutoff = np.quantile(vals, sig_threshold)
            cutoffs[interaction_name] = cutoff

        interaction_to_pathway[interaction_name] = pathway

    if len(activity_matrix_cols) == 0:
        warnings.warn("No significant interactions were found in the dataset")
        return

    activity_matrix = pd.DataFrame(activity_matrix_cols, index=adata.obs.index)

    # Filter out inactive interactions
    filtered_interactions = filtered_interactions[filtered_interactions["interaction_name"].isin(activity_matrix_cols)]
    adata.uns["interactions"] = filtered_interactions

    if perform_permutation:
        adata.uns["activity_significance_cutoff"] = cutoffs
    if save_activity:
        adata.uns['activity_matrix'] = activity_matrix
        adata.obs = pd.concat([adata.obs, activity_matrix], axis=1)

    return activity_matrix


def filter_interactions(interaction, adata, filter_same=True, pathway_filter=None,
                        interaction_filter=None):
    """
    Given a particular adata object, filter out only the LR interactions in the database for which
    both the L and the R are present in the genes in the adata. Modifies the database accordingly.
    """
    st_genes = {v for v in adata.var_names}
    interaction_index = []

    lr_set = set()
    r_set = set()
    pathways = set()

    for i, row in interaction.iterrows():
        pathway = row["pathway_name"]
        if pathway_filter is not None and pathway not in pathway_filter:
            continue

        if interaction_filter is not None and row["interaction_name"] not in interaction_filter:
            continue

        ligand, receptor = row["ligand"], row["receptor"]

        if filter_same and ligand == receptor:
            continue

        if ligand in st_genes and receptor in st_genes:
            lr_set.add(row["interaction_name"])
            r_set.add(receptor)
            pathways.add(row["pathway_name"])
            interaction_index.append(i)

    interaction_index = np.array(interaction_index)
    filtered_interactions = interaction.iloc[interaction_index].reset_index(drop=True)

    adata.uns["interactions"] = filtered_interactions

    if len(filtered_interactions) == 0:
        warnings.warn("No interactions from database matched in dataset.")
        # TODO: remove the debug error
        raise ValueError

    # return the list of identified interactions
    return list(filtered_interactions["interaction_name"])


def filter_adata(adata):
    """Filter out genes that are not part of a known LR interaction (or for which the partner
    is not present)"""
    filtered_interactions = adata.uns["interactions"]
    ligand_genes = np.unique(filtered_interactions["ligand"])
    receptor_genes = np.unique(filtered_interactions["receptor"])
    ligand_receptor_genes = np.unique(np.concatenate((ligand_genes, receptor_genes)))
    gene_names = np.array([v for v in adata.var_names])
    ligand_receptor_gene_inds = np.array([np.where(gene_names == gene)[0]
                                          for gene in ligand_receptor_genes]).flatten()
    adata_filtered = adata[:, ligand_receptor_gene_inds]
    adata_filtered.var_names = gene_names[ligand_receptor_gene_inds]
    return adata_filtered


def compute_active_sub(M, K):
    receptor_active = np.array(np.sum(M, axis=0)).reshape(-1)
    receptor_active = receptor_active / (K + receptor_active)
    return receptor_active


def compute_active_matrix(adata, P, K=0.5, smoothing=0.0, A=None, min_active_count=0):
    # Compute cell-by-active matrix
    active_matrix = np.zeros((len(adata), len(P)), dtype=np.float64)
    columns = []
    col_inds = []
    for k, (interaction_name, M) in enumerate(P.items()):
        v = compute_active_sub(M, K)
        # filter out those below the precomputed significance level
        # +v[v < cutoffs[interaction_name]] = 0
        if np.count_nonzero(v) < min_active_count:
            continue
        columns.append(interaction_name)
        col_inds.append(k)
        active_matrix[:, k] = v
    active_matrix = active_matrix[:, np.array(col_inds)]

    active_matrix = pd.DataFrame(active_matrix, columns=columns)
    return active_matrix, columns


def combine_pathway(filtered_interactions, active_matrix, type="arithmetic"):
    """
    Take an activity matrix and combine all activity on a per-pathway basis
    """

    interactions_by_pathway = {k: [] for k in
                               np.unique(filtered_interactions["pathway_name"])}
    for _, row in filtered_interactions.iterrows():
        interactions_by_pathway[row["pathway_name"]].append(row["interaction_name"])
    res = {}
    if type == "geometric":
        mean = gmean
    elif type == "arithmetic":
        mean = np.mean
    else:
        raise ValueError("Type must be either 'geometric' or 'arithmetic'")
    for pathway, interactions in interactions_by_pathway.items():
        try:
            v = mean(active_matrix.loc[:, interactions], axis=1)
        except KeyError as e:
            print(list(active_matrix))
            raise e
        res[pathway] = v
    active_matrix_pathway = pd.DataFrame.from_dict(res)

    return active_matrix_pathway


def combine_receptor(filtered_interactions, active_matrix, type="arithmetic"):
    """
    Take an activity matrix and combine all activity on a per-receptor basis
    """

    interactions_by_receptor = {k: [] for k in
                                np.unique(filtered_interactions["receptor"])}
    for _, row in filtered_interactions.iterrows():
        interactions_by_receptor[row["receptor"]].append(row["interaction_name"])

    res = {}
    if type == "geometric":
        mean = gmean
    elif type == "arithmetic":
        mean = np.mean
    else:
        raise ValueError("Type must be either 'geometric' or 'arithmetic'")
    for pathway, interactions in interactions_by_receptor.items():
        res[pathway] = mean(active_matrix.loc[:, interactions], axis=1)
    active_matrix_receptor = pd.DataFrame.from_dict(res)

    return active_matrix_receptor


def cell_type_activity(adata, format, type_label="class"):
    """
    Compute the average activity over all
    :param adata: anndata object after running compute_activity
    :param format:
    :param type_label: categorical label in adata.obs that defines cell types (or other exclusive grouping)
    :return:
    """

    # Either process for LR pairs, receptors, or pathways
    if format != "LR":
        raise NotImplementedError

    cols = list(adata.uns["interactions"]["interaction_name"])

    cols = ["activity_%s" % k for k in cols]
    # print(cols)
    active_matrix = adata.obs.loc[:, cols]

    types = np.unique(adata.obs[type_label])
    rows = []
    significant_mat = np.zeros((len(types), active_matrix.shape[1]))

    active_matrix_array = np.array(active_matrix)
    for k, t in enumerate(types):
        # test for significance
        active_matrix_sub = active_matrix_array[(adata.obs[type_label] == t)]
        sub_activity = np.mean(np.array(active_matrix)[(adata.obs[type_label] == t)], axis=0)

        U, p = mannwhitneyu(active_matrix_sub, active_matrix_array,
                            alternative='greater')
        significant = p < 0.05
        rows.append(sub_activity)
        significant_mat[k, :] = significant

    # take out the "activity_"
    cols = [k[9:] for k in active_matrix.columns]
    out = pd.DataFrame(rows, columns=cols, index=types)

    def style_significant(v):
        return np.where(significant_mat, "font-weight: bold", "font-weight: lighter")

    style = out.style.apply(style_significant, axis=None)
    return out, style
