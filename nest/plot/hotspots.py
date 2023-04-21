import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.colors as colors
import pandas as pd
import numpy as np
import seaborn as sns
import colorcet as cc
from nest.plot.plot import spatial, scale


def hotspots(adata, color, ax=None, show=True, correct_categorical=False, labels=True,
             **kwargs):
    # TODO: docstring
    if ax is None:
        ax = plt.gca()
    if "title" not in kwargs:
        kwargs["title"] = " - ".join([v.capitalize() for v in color.split("_")])

    key = "hotspots_%s" % color

    if correct_categorical and not hasattr(adata.obs[key], 'cat'):
        v = pd.Categorical(adata.obs[key])
        v.categories = v.categories.astype(np.int_)
        adata.obs[key] = v

    if labels:
        legend_loc = "on data"
    else:
        legend_loc = None
    spatial(adata, color=key,
            ax=ax, show=False, legend_loc=legend_loc, na_in_legend=False,
            **kwargs)

    if show:
        plt.show()


def hotspots_multiple(adata, nrows, ncols, hotspot_keys=None, figsize=None):

    if hotspot_keys is None:
        obs_keys = list(adata.obs)
        hotspot_keys = [v[9:] for v in obs_keys if "hotspots_" in v]
        nrows = int(np.ceil(len(hotspot_keys) / ncols))

    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)

    for i, hotspot_key in enumerate(hotspot_keys):
        cur_row = i // ncols
        cur_col = i % ncols
        hotspots(adata, hotspot_key, ax=axs[cur_row, cur_col], show=False)

    for i in range(nrows):
        for j in range(ncols):
            axs[i, j].set_xlabel('')
            axs[i, j].set_ylabel('')
    plt.tight_layout()


def multi_hotspots(adata, show=True, inds=None, spotcolor=None, ax=None, num=None, linewidth=0.5,
                   legend_ncol=None, alpha_img=1.0, color_type=None, alpha=150, base_color=None,
                   frameon=False, cm1=None, show_colorbar=False,
                   **kwargs):
    # TODO: docstring
    if color_type is None:
        color_type = "index"

    #cm1 = mpl.cm.get_cmap('tab20')
    if cm1 is None:
        cm1 = sns.color_palette(cc.glasbey_bw, n_colors=len(adata.uns['multi_hotspots']))
    legend_handles = []
    legend_labels = []
    create_legend = True

    if color_type == "genes":
        create_legend = False
        vals = [len(v) for k, v in adata.uns["multi_hotspots"].items()]
        min_genes = min(vals)
        max_genes = max(vals)
        norm = colors.LogNorm(vmin=min_genes, vmax=max_genes)
        cmap = mpl.cm.YlOrRd

        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)


    try:
        scale_factor = list(adata.uns['spatial'].values())[0]['scalefactors']['tissue_hires_scalef']
    except KeyError:
        if scale is not None:
            scale_factor = scale
        else:
            scale_factor = 1

    if ax is None:
        fig, ax = plt.subplots()

    spatial(adata, color=base_color, ax=ax, alpha_img=alpha_img, frameon=frameon,
            na_color=spotcolor,
            show=False, **kwargs)

    hotspot_keys = [v for v in adata.obs if "hotspots_multi" in v]
    if inds is None:
        # Draw boundaries around all considered regions
        inds = list(range(len(hotspot_keys)))
    for c, idx in enumerate(inds):
        if num is not None and idx >= num:
            break
        if color_type == "index":
            try:
                color = adata.uns['multi_hotspots_colors'][idx]
            except (AttributeError, KeyError) as e:
                color = cm1[c]
        elif color_type == "genes":
            color = sm.to_rgba(len(adata.uns["multi_hotspots"][str(idx)]))
        else:
            # assume it is a vector with colors
            color = colors.to_rgba(color_type[idx])

        color_int = [int(255 * v) for v in color]
        color_string = "#%02x%02x%02x%02x" % (color_int[0], color_int[1], color_int[2], alpha)
        color_string_noalpha = "#%02x%02x%02x" % (color_int[0], color_int[1], color_int[2])

        if True or idx < 21:
            legend_labels.append(idx)
            #legend_handles.append(Patch(facecolor=color_string, edgecolor='k',
            #                            label=idx))
            legend_handles.append(Line2D([0], [0], marker='s', color='w', label=idx,
                   markerfacecolor=color_string_noalpha, markersize=8))

        try:
            boundary = scale_factor*adata.uns["multi_boundaries"][str(idx)]
        except KeyError:
            raise ValueError("Need to compute boundaries first")
        except TypeError:
            continue
        ax.fill(boundary[:, 0], boundary[:, 1], color_string)
        ax.plot(boundary[:, 0], boundary[:, 1], linewidth=linewidth, color="black")
    if legend_ncol is not None and create_legend:
        ax.legend(legend_handles, legend_labels, loc="center left", bbox_to_anchor=(1, 0.5),
                  ncol=legend_ncol, frameon=False)
        #ax.legend(legend_handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.0),
        #          ncol=legend_ncol)
    elif color_type == "genes":
        ax.get_figure().colorbar(sm, orientation='vertical', label='coexpressed genes',
                                 shrink=0.8)
    elif show_colorbar == True:
        ax.get_figure().colorbar(mpl.cm.ScalarMappable(norm=colors.NoNorm(vmin=0, vmax=1), cmap=sns.color_palette("Reds", as_cmap=True)), 
                                 orientation='vertical', label="",
                                 shrink=0.6)


    #plt.tight_layout()
    if show:
        plt.show()


def all_coex_hotspots(adata, ncol=4, figsize=None, **kwargs):
    num_coex_hotspot = len(adata.uns['multi_hotspots']) + 1
    nrow = np.ceil(num_coex_hotspot / ncol).astype(np.int_)
    fig, axs = plt.subplots(nrow, ncol, figsize=figsize)
    axs = axs.ravel()
    keyfn = lambda x: int(x[0])
    for idx, (k, multi_hotspot) in enumerate(sorted(adata.uns['multi_hotspots'].items(), key=keyfn)):
        num_genes = len(multi_hotspot)
        hotspots(adata, f"multi_{k}", title=f"CH {k} ({num_genes} genes)", labels=None, palette="b",
                         alpha_img=0.5, show=False, ax=axs[idx], **kwargs)
    for k in range(num_coex_hotspot-1, nrow*ncol):
        axs[k].set_axis_off()