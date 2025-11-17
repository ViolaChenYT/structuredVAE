import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import scanpy as sc
import scvi
import seaborn as sns
import torch
import anndata
import networkx as nx
import sys
import argparse
from typing import Dict
import gzip
from scvi_celegan_path import train_scvi_on_path

def parse_args():
    p = argparse.ArgumentParser(
        description="Prep all paths"
    )
    p.add_argument("--path_dict", type=str, \
        default = "/n/fs/ragr-data/users/viola/mouse_dev/scripts/new_c_elegans/paths_dict_tree_small.json.gz", \
        help="Input json for path dictionary")
    p.add_argument("--node_abbrev", type=str, \
        default = "/n/fs/ragr-data/users/viola/mouse_dev/scripts/new_c_elegans/original_to_merged.csv", \
        help="Input csv for node abbreviations")
    p.add_argument("--fuzzy_mapping", type=str,\
        default="/n/fs/ragr-data/users/viola/mouse_dev/scripts/new_c_elegans/data/fuzzy_lineage_mapping.json", \
        help="Input json for fuzzy mapping")
    return p.parse_args()

def load_path_dict(path: str) -> Dict[str, dict]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        return json.load(f)
        
def load_json(path: str) -> Dict[str, dict]:
    with open(path, "r") as f:
        return json.load(f)

def train_scvi_for_paths(adata, lineage_paths, base_path="/n/fs/ragr-data/users/viola/structuredVAE/", 
                         group_to_nodes=None, fuzzy_mapping=None):
    """
    Train scVI models for multiple lineage paths using the modular function.
    
    Parameters:
    -----------
    adata : anndata.AnnData
        Input AnnData object containing the full dataset
    lineage_paths : list of lists
        List of lineage path lists, where each path is a list of lineage labels
        (e.g., [["MSxap", "MSxapp", "MSxappp"], ["MSxap", "MSxapp", "MSxappp", "MSxappppx"]])
    base_path : str
        Base path for saving outputs
    group_to_nodes : dict, optional
        Dictionary mapping group names to lists of nodes (default: None)
    fuzzy_mapping : dict, optional
        Dictionary mapping nodes to lists of possible lineage values (default: None)
    
    Returns:
    --------
    dict
        Dictionary mapping path names to training results
    """
    results = {}
    
    for path_name, lineage_path in lineage_paths.items():
        print(f"Training scVI for path: {' -> '.join(lineage_path)}")
        try:
            result = train_scvi_on_path(
                adata=adata,
                lineage_path=lineage_path,
                path_name=path_name,
                base_path=base_path,
                save_loom=True,
                save_model=True,
                save_plots=True,
                fuzzy_mapping=fuzzy_mapping
            )
            
            results[path_name] = result
            print(f"✓ Successfully trained model for {' -> '.join(lineage_path)}")
            
        except Exception as e:
            print(f"✗ Failed to train model for {' -> '.join(lineage_path)}: {str(e)}")
            results[path_name] = None
    
    return results


def main():
    args = parse_args()
    path_dict = load_path_dict(args.path_dict)

    # node_abbrev = pd.read_csv(args.node_abbrev, index_col=0)
    # # Create dictionary mapping merged_group to list of nodes
    # group_to_nodes = {}
    # for node, row in node_abbrev.iterrows():
    #     merged_group = row['merged_group']
    #     if merged_group not in group_to_nodes:
    #         group_to_nodes[merged_group] = []
    #     group_to_nodes[merged_group].append(node)

    fuzzy_mapping = load_json(args.fuzzy_mapping)
    
    # Example usage of the modular function:
    # Load the data
    base_path = "/n/fs/ragr-data/users/viola/structuredVAE/"
    adata = sc.read(f"{base_path}/data/packer2019_preprocessed.h5ad")
    
    example_paths = {"MSxap_MSxappppx": ["MSxap", "MSxapp", "MSxappp", "MSxappppx"],}
    # result = train_scvi_on_path(adata, example_paths["MSxap_MSxappppx"], "MSxap_MSxappppx", base_path, fuzzy_mapping=fuzzy_mapping)
    # Train models for all paths with fuzzy mapping
    print("Training with fuzzy mapping...")
    results = train_scvi_for_paths(adata, path_dict, base_path, fuzzy_mapping=fuzzy_mapping)
    
    print(f"Training completed for {len([r for r in results.values() if r is not None])} paths")
    
    # print("\nTraining without fuzzy mapping...")
    # results_no_fuzzy = train_scvi_for_paths(adata, example_paths, base_path)
    # print(f"Training completed for {len([r for r in results_no_fuzzy.values() if r is not None])} paths")


if __name__ == "__main__":
    main()