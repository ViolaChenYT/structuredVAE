#!/usr/bin/env python3
"""
Helper script to extract path names from loom files for Snakemake workflow.
This script can be used to generate a list of all available paths.
"""

import os
import glob
import argparse
import json
from pathlib import Path

def get_path_names(data_dir="data"):
    """
    Extract path names from loom files in the data directory.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing loom files
        
    Returns:
    --------
    list
        List of path names (without .loom extension)
    """
    loom_files = glob.glob(os.path.join(data_dir, "*.loom"))
    path_names = []
    
    for loom_file in loom_files:
        # Extract path name from filename (remove .loom extension)
        path_name = os.path.basename(loom_file).replace(".loom", "")
        path_names.append(path_name)
    
    return sorted(path_names)

def check_scvae_results(data_dir="data", path_name=None):
    """
    Check if scVAE results exist for a given path.
    
    Parameters:
    -----------
    data_dir : str
        Base data directory
    path_name : str
        Path name to check
        
    Returns:
    --------
    bool
        True if scVAE results exist, False otherwise
    """
    if path_name is None:
        return False
        
    # Expected scVAE results directory structure
    scvae_dir = os.path.join(data_dir, f"scvae_path_{path_name}", path_name)
    expected_files = [
        "no_split/no_preprocessing/GMVAE/gaussian_mixture-c_6/zero_inflated_negative_binomial-l_1-h_64_64-mc_1-iw_1-bn-wu_200/e_500-mc_1-iw_1/full/latent_values-y.tsv.gz",
        "no_split/no_preprocessing/GMVAE/gaussian_mixture-c_6/zero_inflated_negative_binomial-l_1-h_64_64-mc_1-iw_1-bn-wu_200/e_500-mc_1-iw_1/full/latent_values-z.tsv.gz"
    ]
    
    for expected_file in expected_files:
        file_path = os.path.join(scvae_dir, expected_file)
        if not os.path.exists(file_path):
            return False
    
    return True

def get_available_paths(data_dir="data"):
    """
    Get list of paths that have both loom files and scVAE results.
    
    Parameters:
    -----------
    data_dir : str
        Base data directory
        
    Returns:
    --------
    dict
        Dictionary with available paths and their status
    """
    all_paths = get_path_names(data_dir)
    available_paths = {
        "total_paths": len(all_paths),
        "paths_with_scvae_results": [],
        "paths_without_scvae_results": [],
        "all_paths": all_paths
    }
    
    for path_name in all_paths:
        if check_scvae_results(data_dir, path_name):
            available_paths["paths_with_scvae_results"].append(path_name)
        else:
            available_paths["paths_without_scvae_results"].append(path_name)
    
    return available_paths

def main():
    parser = argparse.ArgumentParser(description="Extract path names from loom files")
    parser.add_argument("--data_dir", default="data", help="Data directory containing loom files")
    parser.add_argument("--output", help="Output file to save path names (JSON format)")
    parser.add_argument("--check_scvae", action="store_true", help="Check which paths have scVAE results")
    parser.add_argument("--available_only", action="store_true", help="Only return paths with scVAE results")
    
    args = parser.parse_args()
    
    if args.check_scvae:
        available_paths = get_available_paths(args.data_dir)
        
        print(f"Total paths found: {available_paths['total_paths']}")
        print(f"Paths with scVAE results: {len(available_paths['paths_with_scvae_results'])}")
        print(f"Paths without scVAE results: {len(available_paths['paths_without_scvae_results'])}")
        
        if args.available_only:
            path_names = available_paths["paths_with_scvae_results"]
        else:
            path_names = available_paths["all_paths"]
    else:
        path_names = get_path_names(args.data_dir)
    
    if args.output:
        output_data = {
            "path_names": path_names,
            "data_dir": args.data_dir,
            "total_count": len(path_names)
        }
        
        if args.check_scvae:
            output_data.update(available_paths)
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Path names saved to {args.output}")
    else:
        for path_name in path_names:
            print(path_name)

if __name__ == "__main__":
    main()
