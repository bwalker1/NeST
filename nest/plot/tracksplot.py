import pandas as pd
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns


def tracks_plot(adata, marker_genes, width=12, track_height=0.25,
                fontsize=None, use_raw=False, exp_transform=False, labels=None,
                other_genes=None, palette=None, gene_name_italic=False, marked_genes=None):
    # Currently not setting fontsize causes it to not work
    if other_genes is None:
        other_genes = []
    if gene_name_italic:
        italicstyle = 'italic'
    else:
        italicstyle = None
    keys = []
    tracksplot_genes = []
    categories = []
    category_labels = []
    label_rotation = 0
    for k, genes in marker_genes.items():
        categories.append(k)
        if labels is not None:
            label = labels[k]
        else:
            label = f"CH{k}"
        category_labels.append(label)

        keys.append(f'hotspots_multi_{k}')

        for gene in genes:
            if gene not in keys:
                keys.append(gene)
                tracksplot_genes.append(gene)

    for gene in other_genes:
        if gene not in keys:
            keys.append(gene)
            tracksplot_genes.append(gene)


    # Prepare the data for the tracksplot
    obs_tidy = sc.get.obs_df(adata, keys=keys, use_raw=use_raw)

    # Create the tracksplot
    nbins = 10

    dendro_height = 0

    groupby_height = 0.24
    # +2 because of dendrogram on top and categories at bottom
    num_rows = len(tracksplot_genes)
    #width = 12
    #track_height = 0.25

    height_ratios = [track_height] * len(tracksplot_genes)
    height = 2 * sum(height_ratios)

    # TODO: sort out colors
    groupby_colors = None
    if palette is not None:
        groupby_colors = sns.color_palette(palette)[:len(categories)]

    if groupby_colors is None and len(categories) <= 10:
        groupby_colors = sns.color_palette("deep")[:len(categories)]

    fig = plt.figure(figsize=(width, height))
    axs = gridspec.GridSpec(
        ncols=2,
        nrows=num_rows,
        wspace=1.0 / width,
        hspace=0,
        height_ratios=height_ratios,
        width_ratios=[width, 0.14],
    )
    axs_list = []
    first_ax = None
    for idx, gene in enumerate(tracksplot_genes):
        ax_idx = idx
        if first_ax is None:
            ax = fig.add_subplot(axs[ax_idx, 0])
            first_ax = ax
        else:
            ax = fig.add_subplot(axs[ax_idx, 0], sharex=first_ax)
        axs_list.append(ax)
        for cat_idx, category in enumerate(categories):
            #obs_cur_cat = obs_tidy[obs_tidy["hotspots_%s"%interaction] == category]
            obs_cur_cat = obs_tidy[pd.notnull(obs_tidy[f'hotspots_multi_{category}'])]
            expression_values = np.sort(
                obs_cur_cat.loc[:, gene].to_numpy())  # Get the expression_values
            if exp_transform:
                expression_values = np.exp(expression_values)-1
            average_expressions = np.zeros(nbins)

            num = int(np.floor(np.size(expression_values) / nbins))

            for ave_idx in range(nbins):
                if ave_idx < nbins - 1:
                    try:
                        average_expressions[ave_idx] = np.mean(
                            expression_values[num * (ave_idx):num * (1 + ave_idx)])
                    except ZeroDivisionError:
                        # TODO: this probably isn't necessary
                        print(expression_values.shape)
                        print(num * (ave_idx), num * (1 + ave_idx))
                else:
                    average_expressions[ave_idx] = np.mean(expression_values[num * (ave_idx):])

            if groupby_colors is not None:
                cc = groupby_colors[cat_idx]
            else:
                cc = None
            ax.fill_between(
                range(cat_idx * nbins, (cat_idx + 1) * nbins),
                0,
                average_expressions,
                lw=0.1,
                color=cc,
            )

        # remove the xticks labels except for the last processed plot.
        # Because the plots share the x axis it is redundant and less compact
        # to plot the axis for each plot
        if idx < len(tracksplot_genes) - 1:
            try:
                ax.tick_params(labelbottom=False, labeltop=False, bottom=False, top=False,
                               size=fontsize)
                ax.set_xlabel('')
            except TypeError:
                # this fails safely in jupyter notebooks
                pass
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.grid(False)
        ymin, ymax = ax.get_ylim()
        ymaxlabel = int(np.ceil(ymax))
        #ax.set_yticks([ymax])
        ax.set_yticks([])
        #ax.set_yticklabels([str(ymaxlabel)], ha='left', va='top', fontsize=fontsize)
        ax.spines['right'].set_position(('axes', 1.01))
        ax.tick_params(
            axis='y',
            labelsize=fontsize,
            right=True,
            left=False,
            length=2,
            which='both',
            labelright=True,
            labelleft=False,
            direction='in',
            #size=fontsize,
        )
        if gene in marked_genes:
            gene_label = f"{gene}*"
        else:
            gene_label = gene
        ax.set_ylabel(gene_label, rotation=0, fontsize=fontsize, ha='right', va='bottom',
                      style=italicstyle)
        ax.yaxis.set_label_coords(-0.005, 0.1)
    ax.set_xlim(0, len(categories) * nbins)
    #ax.tick_params(axis='x', bottom=False, labelbottom=False)
    ax.set_xticks(nbins * (np.arange(len(categories))+0.5))
    ax.set_xticklabels(category_labels, ha='left', va='top', rotation=label_rotation,
                       fontsize=fontsize)
    # groupby_ax.tick_params(bottom=True)
    if type=="hotspot":
        ax.set_xlabel('Hotspot', fontsize=fontsize)
        fig.suptitle(' - '.join([v.capitalize() for v in interaction.split('_')]), fontsize=fontsize, y=0.95)

    # the ax to plot the groupby categories is split to add a small space between the rest of the plot and the categories
    #axs2 = gridspec.GridSpecFromSubplotSpec(
    #    2, 1, subplot_spec=axs[num_rows - 1, 0], height_ratios=[1, 1]
    #)

    #groupby_ax = fig.add_subplot(axs2[1])
    #groupby_ax.axis('off')

    #fig.tight_layout()
    return fig, ax


if __name__ == "__main__":
    import nest
    import os
    available_datasets = ["V1_Mouse_Brain_Sagittal_Anterior", "V1_Mouse_Brain_Sagittal_Posterior",
                          "seqfish", "merfish",
                          "slideseq", "V1_Breast_Cancer_Block_A_Section_1"]
    dataset = "V1_Breast_Cancer_Block_A_Section_1"
    cache_dir = os.path.expanduser(f"~/Dropbox/data/ms/datasets/{dataset}")
    adata = nest.read(os.path.join(cache_dir, 'adata'))
    markers = nest.geometric_markers(adata, [0, 1, 2, 5])
    nest.plot.tracks_plot(adata, markers)
    plt.show()