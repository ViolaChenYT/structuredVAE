import os
import argparse
import json
import gzip
import subprocess
from typing import Dict

def parse_args():
    p = argparse.ArgumentParser(
        description="Train scVAE on lineage paths using existing loom files from path_dict"
    )
    p.add_argument("--path_dict", type=str, \
        default = "/n/fs/ragr-data/users/viola/mouse_dev/scripts/new_c_elegans/paths_dict_tree.json.gz", \
        help="Input json for path dictionary")
    return p.parse_args()

def load_path_dict(path: str) -> Dict[str, dict]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        return json.load(f)
        

def train_scvae_on_path(lineage_path, path_name, base_path="/n/fs/ragr-data/users/viola/structuredVAE/"):
    """
    Train scVAE model on a lineage path using existing loom file.
    
    Parameters:
    -----------
    lineage_path : list
        List of lineage labels representing the complete path
    path_name : str
        Name for the path (used for file naming)
    base_path : str
        Base path for saving outputs
    
    Returns:
    --------
    dict
        Dictionary containing training results and paths
    """
    # Calculate number of nodes for scVAE (K is the length of the path itself)
    n_nodes = len(lineage_path)
    
    # Set up file paths
    input_folder = f"{base_path}/data/"
    loom_file = input_folder + path_name + ".loom"
    output_folder = input_folder + "scvae_path_" + path_name + "/"
    
    # Check if loom file exists
    if not os.path.exists(loom_file):
        raise FileNotFoundError(f"Loom file not found: {loom_file}")
    
    # Check if output directory already exists
    if os.path.exists(output_folder):
        print(f"Output directory {output_folder} already exists. Skipping path {path_name}.")
        return {
            'input_file': loom_file,
            'output_folder': output_folder,
            'n_nodes': n_nodes,
            'lineage_path': lineage_path,
            'skipped': True
        }
    
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"Training scVAE for path: {' -> '.join(lineage_path)}")
    print(f"Number of nodes (K): {n_nodes}")
    print(f"Using loom file: {loom_file}")
    
    # Train scVAE
    print("training scvae...")
    cmd = f"scvae train {loom_file} -m GMVAE -r zero_inflated_negative_binomial -l 1 -H 64 64 -K {n_nodes} -w 80 -e 400 --batch-correction --models-directory {output_folder}"
    print(cmd)
    os.system(cmd)
    
    # Evaluate scVAE
    print("evaluating scvae...")
    cmd = f"scvae evaluate {loom_file} -m GMVAE -r zero_inflated_negative_binomial -l 1 -H 64 64 -K {n_nodes} -w 80 --batch-correction --models-directory {output_folder} --analyses-directory {output_folder}"
    print(cmd)
    os.system(cmd)
    
    return {
        'input_file': loom_file,
        'output_folder': output_folder,
        'n_nodes': n_nodes,
        'lineage_path': lineage_path,
        'skipped': False
    }

def train_scvae_for_paths(path_dict, base_path="/n/fs/ragr-data/users/viola/structuredVAE/"):
    """
    Train scVAE models for multiple lineage paths using existing loom files.
    
    Parameters:
    -----------
    path_dict : dict
        Dictionary mapping path names to lineage path lists
    base_path : str
        Base path for saving outputs
    
    Returns:
    --------
    dict
        Dictionary mapping path names to training results
    """
    results = {}
    
    for path_name, lineage_path in path_dict.items():
        print(f"Processing path: {path_name}")
        try:
            result = train_scvae_on_path(
                lineage_path=lineage_path,
                path_name=path_name,
                base_path=base_path
            )
            
            results[path_name] = result
            if result.get('skipped', False):
                print(f"⏭ Skipped path {path_name} (directory already exists)")
            else:
                print(f"✓ Successfully trained scVAE model for {path_name}")
            
        except Exception as e:
            print(f"✗ Failed to train scVAE model for {path_name}: {str(e)}")
            results[path_name] = None
    
    return results

def main():
    args = parse_args()
    path_dict = load_path_dict(args.path_dict)
    
    # Set base path
    base_path = "/n/fs/ragr-data/users/viola/structuredVAE/"
    
    # Train scVAE models for all paths using existing loom files
    print("Training scVAE using existing loom files...")
    results = train_scvae_for_paths(path_dict, base_path)
    
    # Count successful and skipped results
    successful = len([r for r in results.values() if r is not None and not r.get('skipped', False)])
    skipped = len([r for r in results.values() if r is not None and r.get('skipped', False)])
    failed = len([r for r in results.values() if r is None])
    
    print(f"Training completed:")
    print(f"  - Successfully trained: {successful} paths")
    print(f"  - Skipped (already exists): {skipped} paths")
    print(f"  - Failed: {failed} paths")

if __name__ == "__main__":
    main()