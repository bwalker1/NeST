import nest
import os
import anndata
import scipy
import sklearn.metrics
import numpy as np
import pandas as pd
import seaborn as sns
import scanpy as sc
import squidpy as sq
import networkx as nx
from pathlib import Path
import time
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rc
import networkx as nx
import colorcet as cc

import warnings
warnings.filterwarnings("ignore")


def load_dataset():
    dataset = "slideseq"
    neighbor_eps = 75
    min_samples = 5
    hotspot_min_size = 50

    adata = nest.data.get_data(dataset, normalize=True)

    # Note: we perform this spatial smoothing for this specific dataset
    nest.spatial_smoothing(adata)






if __name__=="__main__":
    # Load and process slideseq data
    pass