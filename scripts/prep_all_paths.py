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
        default = "/n/fs/ragr-data/users/viola/mouse_dev/scripts/new_c_elegans/paths_dict_min15.json.gz", \
        help="Input json for path dictionary")
    p.add_argument("--node_abbrev", type=str, \
        default = "/n/fs/ragr-data/users/viola/mouse_dev/scripts/new_c_elegans/original_to_merged.csv", \
        help="Input csv for node abbreviations")
    p.add_argument("--fuzzy_mapping", type=str,\
        default="/n/fs/ragr-data/users/viola/mouse_dev/scripts/new_c_elegans/data/fuzzy_lineage_mapping.json", \
        help="Input json for fuzzy mapping")
    p.add_argument("--remake-plots-only", action="store_true",
                   help="If set, remake plots from existing trained.h5ad files. If trained.h5ad doesn't exist, train normally.")
    p.add_argument("--remake-reconstruction-plots-only", action="store_true",
                   help="If set, only remake reconstruction loss plots from existing training_history.csv files.")
    return p.parse_args()

def load_path_dict(path: str) -> Dict[str, dict]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        return json.load(f)
        
def load_json(path: str) -> Dict[str, dict]:
    with open(path, "r") as f:
        return json.load(f)

def check_output_exists(path_name, base_path):
    """
    Check if outputs for a given path already exist by checking for trained.h5ad file.
    
    Parameters:
    -----------
    path_name : str
        Name for the path
    base_path : str
        Base path for outputs
    
    Returns:
    --------
    bool
        True if trained.h5ad exists, False otherwise
    """
    model_path = f"{base_path}/data/scvi_path_{path_name}/"
    trained_h5ad = os.path.join(model_path, 'trained.h5ad')
    return os.path.exists(trained_h5ad)


def train_scvi_for_paths(adata, lineage_paths, base_path="/n/fs/ragr-data/users/viola/structuredVAE/", 
                         group_to_nodes=None, fuzzy_mapping=None, remake_plots_only=False, 
                         remake_reconstruction_plots_only=False):
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
    remake_plots_only : bool, optional
        If True, remake plots from existing trained.h5ad files. If trained.h5ad doesn't exist, train normally (default: False)
    remake_reconstruction_plots_only : bool, optional
        If True, only remake reconstruction loss plots from existing training_history.csv files (default: False)
    
    Returns:
    --------
    dict
        Dictionary mapping path names to training results
    """
    results = {}
    
    for path_name, lineage_path in lineage_paths.items():
        print(f"Checking path: {' -> '.join(lineage_path)}")
        
        # If remake_plots_only or remake_reconstruction_plots_only is False, check if outputs already exist and skip
        if not remake_plots_only and not remake_reconstruction_plots_only and check_output_exists(path_name, base_path):
            print(f"⏭ Skipping {path_name} - outputs already exist")
            results[path_name] = {'skipped': True, 'path_name': path_name}
            continue
        
        if remake_reconstruction_plots_only:
            print(f"Remaking reconstruction loss plots for path: {' -> '.join(lineage_path)}")
        elif remake_plots_only:
            print(f"Remaking plots for path: {' -> '.join(lineage_path)}")
        else:
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
                fuzzy_mapping=fuzzy_mapping,
                remake_plots_only=remake_plots_only,
                remake_reconstruction_plots_only=remake_reconstruction_plots_only
            )
            
            results[path_name] = result
            if remake_reconstruction_plots_only:
                print(f"✓ Successfully remade reconstruction loss plots for {' -> '.join(lineage_path)}")
            elif remake_plots_only:
                print(f"✓ Successfully remade plots for {' -> '.join(lineage_path)}")
            else:
                print(f"✓ Successfully trained model for {' -> '.join(lineage_path)}")
            
        except Exception as e:
            print(f"✗ Failed to process model for {' -> '.join(lineage_path)}: {str(e)}")
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
    if args.remake_reconstruction_plots_only:
        print("Remaking reconstruction loss plots with fuzzy mapping...")
    elif args.remake_plots_only:
        print("Remaking plots with fuzzy mapping...")
    else:
        print("Training with fuzzy mapping...")
    results = train_scvi_for_paths(adata, path_dict, base_path, fuzzy_mapping=fuzzy_mapping, 
                                   remake_plots_only=args.remake_plots_only,
                                   remake_reconstruction_plots_only=args.remake_reconstruction_plots_only)
    
    print(f"Training completed for {len([r for r in results.values() if r is not None])} paths")
    
    # print("\nTraining without fuzzy mapping...")
    # results_no_fuzzy = train_scvi_for_paths(adata, example_paths, base_path)
    # print(f"Training completed for {len([r for r in results_no_fuzzy.values() if r is not None])} paths")


if __name__ == "__main__":
    main()