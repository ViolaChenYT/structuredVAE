import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import plotnine as p9
import scanpy as sc
import scvi
import seaborn as sns
import torch
import anndata

adata_path = "/n/fs/ragr-data/users/yihangs/Celegan/packer2019/packer2019.h5ad"
adata = sc.read(
    adata_path,
    backup_url="https://github.com/Munfred/wormcells-site/releases/download/packer2019/packer2019.h5ad",
)

adata.raw = adata.copy() 
adata_copy = adata.copy()
torch.set_float32_matmul_precision("high")

sc.pp.normalize_total(adata_copy)
sc.pp.log1p(adata_copy)
sc.pp.highly_variable_genes(adata_copy, n_top_genes=2000, batch_key="batch", subset=True)

adata = adata[:, adata_copy.var_names].copy()
adata.layers["counts"] = adata.X.copy().tocsr()

adata.write_h5ad("packer2019_preprocessed.h5ad")