import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

import nest


def filter_param(df, param):
    filter_set = {1, 2, 3, 4} - {param}
    return df[~df.varied.isin(filter_set)]


if __name__ == "__main__":
    multi_dfs = []
    for spatial_genes in [24, 32, 64, 256, 512]:
        dfs = [pd.read_csv(f'data/synthetic_hierarchy/results_{spatial_genes}_{trial}.csv') for trial in range(1)]
        df = pd.concat(dfs)
        df['spatial_genes'] = spatial_genes
        multi_dfs.append(df)

    combined_df = pd.concat(multi_dfs)

    combined_df['spatial_genes'] = pd.Categorical(combined_df['spatial_genes'])

    sns.set(style="whitegrid", font_scale=0.75)
    
    fig, axs = plt.subplots(1, 4, figsize=[6, 2])

    sns.lineplot(filter_param(combined_df, 0), x="eps", y="score", hue="spatial_genes", ax=axs[0], legend=None)
    axs[0].set_xlim(left=0.05, right=0.2)
    axs[0].set_xlabel('ε')
    axs[0].set_ylabel('Jaccard similarity')

    sns.lineplot(filter_param(combined_df, 1), x="density", y="score", hue="spatial_genes", ax=axs[1], legend=None)
    #axs[0].set_xlim(left=0.05, right=0.2)
    axs[1].set_ylabel('')

    sns.lineplot(filter_param(combined_df, 2), x="threshold", y="score", hue="spatial_genes", ax=axs[2], legend=None)
    #axs[0].set_xlim(left=0.05, right=0.2)
    axs[2].set_ylabel('')

    sns.lineplot(filter_param(combined_df, 3), x="resolution", y="score", hue="spatial_genes", ax=axs[3])
    #axs[0].set_xlim(left=0.05, right=0.2)
    axs[3].set_ylabel('')
    axs[3].legend(title="spatial genes")
    sns.move_legend(axs[3], "center left", bbox_to_anchor=(1, 0.5))

    for k in range(4):
        if k != 0:
            axs[k].set_ylabel('')
            axs[k].axes.yaxis.set_ticklabels([])
        axs[k].set_ylim([0, 1])



    
    fig.savefig(os.path.expanduser('~/Desktop/synthetic_hierarchy.pdf'), dpi=300, transparent=True, bbox_inches="tight")
    plt.show()