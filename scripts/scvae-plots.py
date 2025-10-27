import pandas as pd
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
import argparse

def parse_args():
    p = argparse.ArgumentParser(
        description="Train scVI on a lineage path subset between start and end prefixes."
    )
    p.add_argument("--input_path", type=str, help="Input path for latent value files")
    p.add_argument("--loom_file", type=str, help="Input loom file for lineage data")
    return p.parse_args()


def main():
    args = parse_args()
    input_path = args.input_path
    loom_file = args.loom_file
    #input_path = "/n/fs/ragr-data/users/yihangs/Celegan/structuredVAE/data/scvae_path_MSxap_to_MSxappppx-2/MSxap_to_MSxappppx-2/no_split/no_preprocessing/GMVAE/gaussian_mixture-c_5/zero_inflated_negative_binomial-l_1-h_64_64-mc_1-iw_1-bn-wu_200/e_500-mc_1-iw_1/full/"

    latent_y = pd.read_csv(input_path + "latent_values-y.tsv.gz", sep="\t")
    latent_z = pd.read_csv(input_path + "latent_values-z.tsv.gz", sep="\t")
    latent_z_arr = latent_z["z variable 1"].to_numpy()

    #start = "MSxap"
    #end = "MSxappppx"
    #lineage_path = [end[:len(start)+i] for i in range(0,len(end)-len(start)+1)]

    #adata = sc.read("/n/fs/ragr-data/users/yihangs/Celegan/packer2019/packer2019.h5ad")

    #loom_file = "/n/fs/ragr-data/users/yihangs/Celegan/structuredVAE/data/MSxap_to_MSxappppx-2.loom"
    #adata_lineagepath = adata[adata.obs["lineage"].isin(lineage_path)]
    adata_lineagepath = sc.read_loom(loom_file)

    plt.hist(latent_z_arr, bins=50, edgecolor='black')  # adjust bins as needed
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Frequency Distribution")
    plt.savefig(input_path + "1dlatent_frequency_distribution.png")
    #plt.show()
    plt.close()

    labels = adata_lineagepath.obs["lineage"].tolist()
    unique_labels = np.unique(labels)

    plt.figure(figsize=(7,5))
    for lab in unique_labels:
        subset = latent_z_arr[np.array(labels) == lab]
        plt.hist(subset, bins=50, density=True, alpha=0.5, label=str(lab))

    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Frequency Distributions by Label (normalized)")
    plt.legend()
    #plt.show()
    plt.savefig(input_path+"1dlatent_frequency_distribution_by_lineage.png")
    plt.close()

if __name__ == "__main__":
    main()