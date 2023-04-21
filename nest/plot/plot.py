import os

import matplotlib.pyplot as plt
from matplotlib import colors
import scanpy as sc
import seaborn as sns
import numpy as np
import pandas as pd
import colorcet as cc

# Global variables store dataset-specific information
from nest.hotspot.geometry import similarity_map

library_id = None
img = None
scale = None
spot_size_preset = None
parameters_set = False
image_path = None


def set_dataset_plot_parameters(dataset):
    """
    Sets global variables that are used in spatial plots for non-Visium datasets
    :param dataset:
    :return:
    """
    global library_id, img, scale, spot_size_preset, parameters_set, image_path
    if dataset == "seqfish":
        img = None
        scale = None
        spot_size_preset = 0.045
    elif "merfish" in dataset:
        img = None
        scale = None
        spot_size_preset = 0.02
    elif "imc" in dataset:
        img = None
        scale = None
        spot_size_preset = 10
    elif "slideseq" in dataset:
        img = None
        scale = None
        spot_size_preset = 30
    elif "Spatial_LIBD" in dataset:
        # TODO: load the image
        img = None
        scale = None
        spot_size_preset = 100
    else:
        pass

    # So that we can check that the globals are set correctly later
    parameters_set = True


def spatial(adata, spot_size=None, **kwargs):
    """
    Wraps scanpy's pl.spatial function, using parameters from set_dataset_plot_parameters,
    in order to make it a bit
    less annoying to manage parameters when analyzing non-Visium data
    :param adata: adata object to plot (spatial information in `.obsm['spatial']`)
    :param spot_size: size of spots used to show color parameter
    :param kwargs: Other keyword arguments to pass to sc.pl.spatial, such as color=, etc.
    :return:
    """

    if spot_size is None:
        spot_size = spot_size_preset

    sc.pl.spatial(adata, img=img, scale_factor=scale, spot_size=spot_size,
                  **kwargs)

    try:
        ax = kwargs['ax']
        ax.set_xlabel('')
        ax.set_ylabel('')
    except (KeyError, AttributeError):
        pass


def volcano(df, interaction=None, gene_list=None, ax=None, show=True, xlim=None, ylim=None,
            axis_labels=True, fc_cutoff=0, p_cutoff=2, fontsize=None, xoffset=0, yoffset=0.5,
            marker=None, title=None,
            **kwargs):
    if interaction is not None:
        # filter out the actual genes from the interaction
        ligand, receptor = interaction.split("_")
        ligand = ligand.capitalize()
        receptor = receptor.capitalize()
        df = df.drop(index=ligand).drop(index=receptor)
    x = df["log2(fc)"]
    y = df["-log10(p)"]

    # for coloring points based on significance
    hue = np.zeros(shape=x.shape)
    hue[np.logical_and(x > fc_cutoff, y > p_cutoff)] = 1
    hue[np.logical_and(x < -fc_cutoff, y > p_cutoff)] = 2
    palette = []
    if 0 in hue:
        palette.append(sns.color_palette("muted")[7])
        # palette.append("gray")
    if 1 in hue:
        palette.append(sns.color_palette("muted")[3])
        # palette.append("red")
    if 2 in hue:
        palette.append(sns.color_palette("muted")[0])
        # palette.append("blue")

    # For indicating a subset of genes using a different marker
    style = None
    if marker is not None:
        style = np.zeros(len(x), dtype=np.int32)
        for idx in range(len(x)):
            style[idx] = df.index[idx] in marker

    ax = sns.scatterplot(x=x, y=y, hue=hue, palette=palette, legend=None, ax=ax, style=style,
                         markers=['o', '^'],
                         **kwargs)

    for i, row in df.iterrows():
        x = row["log2(fc)"]
        y = row["-log10(p)"]
        if gene_list is not None and (gene_list is True or i in gene_list):
            # gene_list can be either a list or a dictionary containing offsets
            try:
                gene_xoffset, gene_yoffset = gene_list[i]
            except TypeError:
                gene_xoffset, gene_yoffset = xoffset, yoffset
            ax.text(x + gene_xoffset, y + gene_yoffset, i, fontsize=fontsize,
                    horizontalalignment='right')

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    if not axis_labels:
        ax.set_xlabel('')
        ax.set_ylabel('')
    else:
        ax.set_xlabel('log2(fc)', size=fontsize)
        ax.set_ylabel('-log10(p)', size=fontsize)
    plt.xticks(size=fontsize)
    plt.yticks(size=fontsize)

    # TODO: titles
    if title is not None:
        ax.set_title(title)
    elif interaction is not None:
        ax.set_title(" - ".join([v.capitalize() for v in interaction.split("_")]))
    if show:
        plt.show()

    return ax


def plot_similarity_map(adata, idx, adata_ref=None, ax=None, linewidth=0.5, linecolor="black", title="",
                        show=False, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    try:
        scale_factor = list(adata.uns['spatial'].values())[0]['scalefactors'][
            'tissue_hires_scalef']
    except KeyError:
        if scale is not None:
            scale_factor = scale
        else:
            scale_factor = 1
    if adata_ref is None or adata_ref is adata:
        try:
            boundary = scale_factor * adata.uns["multi_boundaries"][str(idx)]
            #ax.fill(boundary[:, 0], boundary[:, 1], color_string)
            ax.plot(boundary[:, 0], boundary[:, 1], linewidth=linewidth, color=linecolor)
        except KeyError:
            # if no boundaries are computed, don't draw the boundary
            pass
        

    arr = similarity_map(adata, idx=idx, adata_ref=adata_ref)

    adata.obs['tmp'] = arr
    spatial(adata, color="tmp", alpha_img=0.5, ax=ax, title=title, frameon=False, show=show,
            **kwargs)


# Construct hotspot tree
from matplotlib.patches import Rectangle, FancyBboxPatch, BoxStyle, Circle, Shadow
from matplotlib.lines import Line2D
from scipy.spatial.distance import jaccard
try:
    from functools import cache
except ImportError:
    # Python 3.8 compatibility
    def cache(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper


def nested_structure_plot(adata, child_threshold=0.75, figsize=None, fontsize=None, legend=True, legend_ncol=None,
                          alpha_high=0.8, alpha_low=0.1, ax=None, legend_kwargs=None):
    if legend_kwargs is None:
        legend_kwargs = {}


    overlap, A = compute_ch_overlap_matrix(adata, child_threshold)
    row, col = np.where(A)
    # find top level of res
    top_level = np.array(list(set(range(len(A))) - set(np.where(A)[0])))
    # compute length based on normalized size
    top_level = {k: np.count_nonzero(pd.notnull(adata.obs[f'hotspots_multi_{k}'])) for k in top_level}
    total = np.sum(list(top_level.items()))
    top_level = {k: v / total for k, v in top_level.items()}
    #print(row, col)
    # reorder hotspots by depth to get the right arrangement
    @cache
    def get_depth(idx, start):
        cur_max = start
        for c in row[col == idx]:
            cur_max = max(cur_max, get_depth(c, start + 1))
        return cur_max

    def set_alpha(color, alpha):
        return colors.to_hex(colors.to_rgba(color, alpha=alpha), keep_alpha=True)

    num_elements = {int(k): np.count_nonzero(pd.notnull(adata.obs[f'hotspots_multi_{k}']))
                    for k, v in adata.uns['multi_hotspots'].items()}

    start_x = 0
    for k, v in sorted(top_level.items(), key=lambda x: -get_depth(x[0], 1)):
        color = adata.uns['multi_hotspots_colors'][str(k)]
        width = v
        top_level[k] = [start_x, width, set_alpha(color, alpha_high), True]
        if row[col == k].size > 0:
            addition = 0.01
        else:
            addition = 0
        start_x += width + addition

    # go to next level
    cur_level = top_level
    levels = [top_level]
    were_children = True

    depth_start = 2
    while were_children:
        next_level = {}
        were_children = False
        for k, (start_x, width, color, is_top) in cur_level.items():
            if not is_top:
                next_level[k] = [start_x, width, set_alpha(color, alpha_low), False]
                continue
            children = row[col == k]
            remaining_frac = 1

            start_x_sub = start_x
            for c in sorted(children, key=lambda x: -get_depth(x, depth_start)):
                were_children = True
                cur_child_frac = overlap[c, k] / num_elements[k]
                remaining_frac -= cur_child_frac
                color_sub = set_alpha(adata.uns['multi_hotspots_colors'][str(c)], 0.8)
                width_sub = cur_child_frac * width
                mul = 1.0
                next_level[c] = [start_x_sub, mul * width_sub, color_sub, True]
                start_x_sub += width_sub

            # draw the remaining bit (possibly the whole bit if no children) in reduced alpha
            if remaining_frac < 1:
                next_level[k] = [start_x_sub, width * remaining_frac, set_alpha(color, alpha_high), False]
            else:
                next_level[k] = [start_x_sub, width * remaining_frac, set_alpha(color, alpha_low), False]



        if not were_children:
            break

        levels.append(next_level)
        cur_level = next_level
        depth_start += 1

    # Bar plot of the coexpression hotspots
    start_y = 1
    height = 0.2

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.set_axis_off()

    for idx, level in enumerate(levels):
        for k, [start_x, width, color, is_top] in level.items():
            # patch = Rectangle([start_x, start_y], width, height, facecolor=color)
            rounding_size = min(0.01, width / 2)
            patch = FancyBboxPatch([start_x, start_y - height], width, height,
                                   boxstyle=BoxStyle("Round",
                                                     pad=-0.001,
                                                     rounding_size=rounding_size),
                                   facecolor=color, edgecolor=[1.0, 1.0, 1.0, 0.0])
            ax.add_patch(patch)
            if False and is_top and width > 0.02:
                ax.text(start_x + width / 2, start_y - 0.101, k, ha='center', va='center',
                        fontdict={'color': 'white', 'size': fontsize})
        ax.text(-0.01, start_y - height / 2, f'Layer {idx + 1}', fontdict={'size': fontsize}, ha='right', va='center')
        start_y -= 1.1 * height

    if legend:
        # TODO: make legend use the fontsize parameter
        legend_handles = []
        for k, color in adata.uns['multi_hotspots_colors'].items():
            legend_handles.append(Circle([0, 0], radius=5, color=color, label=str(k)))
        for k, v in  {'loc': "upper center", 'bbox_to_anchor': (1, 1),
                  'ncol': legend_ncol, 'frameon': False}.items():
            legend_kwargs.setdefault(k, v)

        ax.legend(handles=legend_handles, **legend_kwargs)


def compute_ch_overlap_matrix(adata, threshold, filter_indirect=True):
    num_ch = len(adata.uns['multi_hotspots'])

    # compute the element list for each CH
    element_list = {int(k): np.array(pd.notnull(adata.obs[f'hotspots_multi_{k}']))
                    for k, v in adata.uns['multi_hotspots'].items()}
    num_elements = {k: np.count_nonzero(v) for k, v in element_list.items()}
    overlap = np.zeros(shape=(num_ch, num_ch))
    A = np.zeros(shape=(num_ch, num_ch))
    for idx in range(num_ch):
        for idy in range(num_ch):
            if idx == idy:
                continue
            overlap[idx, idy] = np.count_nonzero(element_list[idx] & element_list[idy])
            A[idx, idy] = overlap[idx, idy] / num_elements[idx]

    A = A > threshold

    if filter_indirect:
        A = A & ~(A @ A)

    return overlap, A