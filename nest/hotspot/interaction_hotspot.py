from tqdm import tqdm
import numpy as np
import pandas as pd

from nest.hotspot.hotspot import _compute_cutoff, compute_hotspots
from nest.utility.utility import get_activity_dataframe
from nest.cci.cci import compute_activity


def interaction_hotspots(adata, verbose=False, interactions=None, save_activity=False,
                         method='permutation', log=False, K=None,
                         min_active_count=1, sig_threshold=0.95, z_key=None, **kwargs):
    # check if activity scores are available
    activity_matrix = get_activity_dataframe(adata)
    if len(list(activity_matrix)) == 0:
        # TODO: really should only compute activity for interactions in the filter if interactions
        # is not None
        try:
            um_scale = adata.uns['um_scale']
        except KeyError:
            # Try to get it automatically assuming Visium data with 55um spots
            try:
                dataset = list(adata.uns['spatial'].keys())[0]
                um_scale = adata.uns['spatial'][dataset]['scalefactors']['spot_diameter_fullres'] / 55
            except KeyError:
                raise ValueError("Please set adata.uns['um_scale'] to the length of one micrometer"
                                 " in the units of adata.obsm['spatial'].") from None

        secreted_std = 50 * um_scale
        contact_threshold = 20 * um_scale

        perform_permutations = method == 'permutation'
        activity_matrix = compute_activity(adata, secreted_std=secreted_std,
                                           contact_threshold=contact_threshold,
                                           sig_threshold=sig_threshold,
                                           perform_permutation=perform_permutations,
                                           save_activity=save_activity, verbose=verbose,
                                           min_active_count=min_active_count,
                                           interactions=interactions,
                                           z_key=z_key, K=K)

    cols = enumerate(list(activity_matrix))

    region_dict = {}

    for _, interaction in cols:
        if interactions is not None and interaction not in interactions:
            # interaction filter
            continue
        # modify this to account for possibly having to go over multiple z layers
        data = activity_matrix[interaction]
        if method == 'permutation':
            cutoff = adata.uns['activity_significance_cutoff'][interaction]
        else:
            cutoff = _compute_cutoff(data, log=log)
        inds = data > cutoff

        region_offset = 0
        if z_key is None:
            regions = compute_hotspots(adata=adata, input_data=np.where(inds)[0], return_regions=True, **kwargs)
        else:
            regions = -1 * np.ones(adata.shape[0])
            for val in np.unique(adata.obs[z_key]):
                cur_slice_inds = np.logical_and(inds, adata.obs[z_key] == val)
                regions_sub = compute_hotspots(adata=adata, input_data=np.where(cur_slice_inds)[0],
                                               return_regions=True, **kwargs)
                # combine together into one array
                out_inds = np.where(pd.notnull(regions_sub))[0]
                print(out_inds)
                regions[out_inds] = regions_sub[out_inds] + region_offset
                print(regions[regions > 0])
                region_offset += np.max(regions_sub[out_inds])
            regions = pd.Categorical(regions, categories=np.arange(1, np.max(regions) + 1))

        if regions is not None:
            region_dict[f"hotspots_{interaction}"] = regions

    hotspots_df = pd.DataFrame(region_dict, index=adata.obs.index)
    adata.obs = pd.concat([adata.obs, hotspots_df], axis=1)


if __name__ == "__main__":
    import squidpy as sq

    adata = sq.datasets.merfish()
    adata.uns['um_scale'] = 0.001
    adata.obs['z'] = adata.obs.Bregma * 0.01
    bregma_values = pd.unique(adata.obs.Bregma)

    neighbor_eps = 0.06
    min_samples = 5
    hotspot_min_size = 5

    adata = adata[:, ["Cck", "Cckar"]].copy()

    n_regions = interaction_hotspots(adata, eps=neighbor_eps, min_size=hotspot_min_size, min_samples=min_samples,
                         core_only=False, method="permutation", z_key='z', verbose=True)
    print(n_regions)
