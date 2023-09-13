import nest
import os
import numpy as np
import pandas as pd
import scanpy as sc
from tqdm import tqdm
from sklearn.metrics import jaccard_score
from collections import defaultdict
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rc

# rc('font', **{'family': 'serif', 'serif': ['Arial'], 'size': 5})
import seaborn as sns


def generate_scores(adata):
    # Run a bunch of HMRF partitions for various values of number of communities
    # k_range = [5, 10, 15, 20, 25]
    k_range = range(2, 31)
    targets = [0, 2, 4, 5, 6, 8, 16, 20]
    methods = ["hmrf", "bayesspace", "spagcn"]
    scores = {k: defaultdict(list) for k in methods}
    for k in tqdm(k_range):
        spagcn = nest.methods.SpaGCN(regions=k)
        spagcn.fit(adata, verbose=False)

        bayesspace = nest.methods.BayesSpace(regions=k)
        bayesspace.fit(adata)
        # exit(0)

        hmrf = nest.hmrf.HMRFSegmentationModel(
            adata, regions=k, eps=180, label_name="class_hmrf", beta=1
        )
        hmrf.fit(max_iterations=200, update_labels=True, verbose=False)

        # find the region with the greatest overlap with each target CH
        partitions = {
            "hmrf": adata.obs["class_hmrf"],
            "bayesspace": adata.obs["class_bayesspace"],
            "spagcn": adata.obs["class_spagcn"],
        }
        for method, partition in partitions.items():
            for target in targets:
                target_membership = pd.notnull(
                    adata.obs[f"hotspots_multi_{target}"]
                ).to_numpy()
                max_score = -1
                max_partition = None
                for val in np.unique(partition):
                    score = jaccard_score(target_membership, partition == val)
                    if score > max_score:
                        max_score = score
                        max_partition = partition
                scores[method][f"ch_{target}"].append((max_score, max_partition))
        with open("test.pkl", "wb") as f:
            pickle.dump(scores, f)


if __name__ == "__main__":
    height = 1.2
    dataset = "V1_Mouse_Brain_Sagittal_Anterior"
    cache_dir = os.path.expanduser(f"data/{dataset}")
    image_save_dir = os.path.expanduser(f"images/{dataset}/")
    if not os.path.exists(image_save_dir):
        os.makedirs(image_save_dir)
    nest.plot.set_dataset_plot_parameters(dataset)
    rc("font", **{"family": "serif", "serif": ["Arial"], "size": 5})
    adata = nest.read(os.path.join(cache_dir, "adata"))
    # adata.layers['exp'] = adata.X.expm1()

    # generate_scores(adata)
    # exit(0)

    with open("data/test.pkl", "rb") as f:
        scores = pickle.load(f)

    chs_to_plot = [0, 2, 6, 8, 16, 20]

    fig, axs = plt.subplots(1, 6, figsize=(6.5, 1.5))
    for idx, ch in enumerate(chs_to_plot):
        ax = axs[idx]
        d = {
            method: [v[0] for v in data[f"ch_{ch}"]] for method, data in scores.items()
        }
        df = pd.DataFrame(d, index=list(range(2, 31)))
        if idx == 5:
            legend = True
        else:
            legend = False
        sns.scatterplot(data=df, ax=ax, legend=legend, markers=True)
        ax.set_ylim([0.0, 1.0])
        ax.set_title(f"CH {ch}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plt.tight_layout()
    fig.savefig(
        os.path.join(image_save_dir, f"comparison.pdf"),
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.show()
