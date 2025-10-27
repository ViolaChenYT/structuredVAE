#!/usr/bin/env python3
"""
Script to train scVAE for a single lineage path.
This script is called by Snakemake for each path in parallel.
"""

import os
import argparse
import json
import gzip
import subprocess
from typing import Dict, List

def parse_args():
    p = argparse.ArgumentParser(
        description="Train scVAE on a single lineage path"
    )
    p.add_argument("--path_name", type=str, required=True,
        help="Name of the path to train")
    p.add_argument("--path_dict",type=str, default = "/n/fs/ragr-data/users/viola/mouse_dev/scripts/new_c_elegans/paths_dict_tree.json.gz",
        help="Path to paths dictionary JSON file")
    p.add_argument("--loom", type=str, required=True,
        help="Path to loom file")
    p.add_argument("--output_folder", type=str, required=True,
        help="Path to output folder")
    return p.parse_args()

def load_path_dict(path: str) -> Dict[str, dict]:
    """Load path dictionary from JSON file."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        return json.load(f)
        
def load_json(path: str) -> Dict[str, dict]:
    """Load JSON file."""
    with open(path, "r") as f:
        return json.load(f)

def train_scvae_on_path(input_file: str, lineage_path: List[str], path_name: str, output_folder: str):
    """
    Train scVAE model on a lineage path subset.
    
    Parameters:
    -----------
    input_file : loom file
        Input object containing the full dataset
    lineage_path : list
        List of lineage labels representing the complete path
    path_name : str
        Name for the path (used for file naming)
    base_path : str
        Base path for saving outputs
    group_to_nodes : dict, optional
        Dictionary mapping group names to lists of nodes (default: None)
    fuzzy_mapping : dict, optional
        Dictionary mapping nodes to lists of possible lineage values (default: None)
    
    Returns:
    --------
    dict
        Dictionary containing training results and paths
    """
    n_nodes = len(lineage_path)
    
    print(f"Training scVAE for path: {' -> '.join(lineage_path)}")
    print(f"Number of nodes: {n_nodes}")
    
    # Train scVAE
    print("training scvae...")
    cmd = f"scvae --plain train {input_file} -m GMVAE -r zero_inflated_negative_binomial -l 1 -H 64 64 -K {n_nodes} -w 80 -e 400 --batch-correction --models-directory {output_folder}"
    print(cmd)
    os.system(cmd)
    
    # Evaluate scVAE
    print("evaluating scvae...")
    cmd = f"scvae --plain evaluate {input_file} -m GMVAE -r zero_inflated_negative_binomial -l 1 -H 64 64 -K {n_nodes} -w 80 --batch-correction --models-directory {output_folder} --analyses-directory {output_folder}"
    print(cmd)
    os.system(cmd)
    
    # Create completion marker
    completion_file = f"{output_folder}/scvae_training_complete.txt"
    os.makedirs(os.path.dirname(completion_file), exist_ok=True)
    with open(completion_file, 'w') as f:
        f.write(f"scVAE training completed for path: {path_name}\n")
        f.write(f"Lineage path: {' -> '.join(lineage_path)}\n")
        f.write(f"Number of nodes: {n_nodes}\n")
    
    return {
        'input_file': input_file,
        'output_folder': output_folder,
        'n_nodes': n_nodes,
    }

def main():
    args = parse_args()
    
    # Load path dictionary
    path_dict = load_path_dict(args.path_dict)
    
    lineage_path = path_dict[args.path_name]
    
    # Train scVAE for this specific path
    print(f"Training scVAE for path: {args.path_name}")
    print(f"Lineage path: {' -> '.join(lineage_path)}")
    
    try:
        result = train_scvae_on_path(
            input_file=args.loom,
            lineage_path=lineage_path,
            path_name=args.path_name,
            output_folder=args.output_folder
        )
        
        print(f"✓ Successfully trained scVAE model for {args.path_name}")
        print(f"  - Input file: {result['input_file']}")
        print(f"  - Output folder: {result['output_folder']}")
        print(f"  - Number of nodes: {result['n_nodes']}")
        
    except Exception as e:
        print(f"✗ Failed to train scVAE model for {args.path_name}: {str(e)}")
        raise

if __name__ == "__main__":
    main()