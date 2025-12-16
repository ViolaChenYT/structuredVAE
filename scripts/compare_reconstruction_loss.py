"""
Compare reconstruction losses between SCVI and SCVI-WAE models.

For each path in paths_dict, computes:
- SCVI reconstruction loss (from pre-trained model)
- SCVI-WAE reconstruction loss (from saved results)
- Difference: scvi_wae_recon_loss - scvi_recon_loss

Plots the distribution of differences across all paths.
"""
import argparse
import json
import gzip
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import scvi
import torch


def load_path_dict(path: str):
    """Load paths dictionary from JSON file."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        return json.load(f)


def get_scvi_reconstruction_loss(path_name, base_dir, device=None):
    """
    Load SCVI model and compute reconstruction loss.
    
    Parameters
    ----------
    path_name : str
        Name of the path
    base_dir : str
        Base directory containing scvi_path_{path_name} folders
    device : torch.device, optional
        Device to run on
        
    Returns
    -------
    float
        Mean reconstruction loss
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model_path = os.path.join(base_dir, f"data/scvi_path_{path_name}")
    trained_h5ad = os.path.join(model_path, "trained.h5ad")
    
    if not os.path.exists(trained_h5ad):
        raise FileNotFoundError(f"SCVI model not found: {trained_h5ad}")
    
    # Load adata
    adata = sc.read(trained_h5ad)
    
    # Load SCVI model
    try:
        vae = scvi.model.SCVI.load(model_path, adata=adata)
    except Exception as e:
        # Try loading from the directory structure
        if os.path.exists(os.path.join(model_path, "model.pt")):
            vae = scvi.model.SCVI.load(model_path, adata=adata)
        else:
            raise ValueError(f"Could not load SCVI model: {e}")
    
    # Compute reconstruction error
    recon_error = vae.get_reconstruction_error(adata, return_mean=True)
    
    # Extract the reconstruction loss value
    if isinstance(recon_error, dict):
        # Usually the key is "reconstruction_loss"
        if "reconstruction_loss" in recon_error:
            return recon_error["reconstruction_loss"]
        else:
            # Return the first value
            return list(recon_error.values())[0]
    else:
        return float(recon_error)


def get_scvi_wae_reconstruction_loss(path_name, results_dir):
    """
    Get SCVI-WAE reconstruction loss from saved results.
    
    Parameters
    ----------
    path_name : str
        Name of the path
    results_dir : str
        Directory containing scvi_wae results pickle files
        
    Returns
    -------
    float
        Final epoch mean reconstruction loss
    """
    results_pickle = os.path.join(results_dir, f"{path_name}_results.pkl")
    
    if not os.path.exists(results_pickle):
        raise FileNotFoundError(f"SCVI-WAE results not found: {results_pickle}")
    
    # Load results
    with open(results_pickle, "rb") as f:
        results = pickle.load(f)
    
    # Get losses_history
    if "losses_history" not in results or not results["losses_history"]:
        raise ValueError(f"No losses_history found in {results_pickle}")
    
    losses_history = results["losses_history"]
    
    # Get final epoch reconstruction loss
    final_losses = losses_history[-1]
    
    if "ae_recon_loss" in final_losses:
        return final_losses["ae_recon_loss"]
    elif "recon_loss" in final_losses:
        return final_losses["recon_loss"]
    else:
        raise ValueError(f"Could not find reconstruction loss in {results_pickle}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare reconstruction losses between SCVI and SCVI-WAE"
    )
    parser.add_argument(
        "--path_dict",
        type=str,
        default="/n/fs/ragr-data/users/viola/mouse_dev/scripts/new_c_elegans/paths_dict_min15.json.gz",
        help="Input JSON file for path dictionary"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/n/fs/ragr-data/users/viola/structuredVAE",
        help="Base directory containing data/scvi_path_* folders"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="/n/fs/ragr-data/users/viola/structuredVAE/results/scvi_wae_results",
        help="Directory containing SCVI-WAE results pickle files"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="/n/fs/ragr-data/users/viola/structuredVAE/results/reconstruction_loss_comparison.csv",
        help="Output CSV file path"
    )
    parser.add_argument(
        "--output_plot",
        type=str,
        default="/n/fs/ragr-data/users/viola/structuredVAE/results/reconstruction_loss_comparison.png",
        help="Output plot file path"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Check if results CSV already exists
    if os.path.exists(args.output_csv):
        print(f"Found existing results file: {args.output_csv}")
        print("Loading existing results...")
        df = pd.read_csv(args.output_csv, index_col="path_name")
        print(f"Loaded {len(df)} paths from existing file")
    else:
        # Load paths dictionary
        path_dict = load_path_dict(args.path_dict)
        
        # Store results
        results_list = []
        
        print("Computing reconstruction losses for all paths...")
        for path_name in path_dict:
            print(f"  Processing {path_name}...")
            try:
                # Get SCVI reconstruction loss
                scvi_recon_loss = get_scvi_reconstruction_loss(
                    path_name, args.base_dir
                )
                
                # Get SCVI-WAE reconstruction loss
                scvi_wae_recon_loss = get_scvi_wae_reconstruction_loss(
                    path_name, args.results_dir
                )
                
                # Compute difference
                diff = scvi_wae_recon_loss - scvi_recon_loss
                
                results_list.append({
                    "path_name": path_name,
                    "scvi_recon_loss": scvi_recon_loss,
                    "scvi_wae_recon_loss": scvi_wae_recon_loss,
                    "difference": diff,
                })
                
                print(f"    SCVI: {scvi_recon_loss:.4f}, SCVI-WAE: {scvi_wae_recon_loss:.4f}, Diff: {diff:.4f}")
                
            except Exception as e:
                print(f"    Error processing {path_name}: {e}")
                import traceback
                traceback.print_exc()
                # Still record with NaN
                results_list.append({
                    "path_name": path_name,
                    "scvi_recon_loss": None,
                    "scvi_wae_recon_loss": None,
                    "difference": None,
                })
        
        # Create DataFrame
        df = pd.DataFrame(results_list)
        df.set_index("path_name", inplace=True)
        
        # Convert any CUDA tensors to CPU and then to native Python types
        for col in df.columns:
            df[col] = df[col].apply(
                lambda x: x.cpu().item() if isinstance(x, torch.Tensor) else x
            )
        
        # Save to CSV
        df.to_csv(args.output_csv)
        print(f"\nResults saved to {args.output_csv}")
    
    # Filter out NaN values for plotting
    df_clean = df.dropna()
    print(df_clean)
    
    if len(df_clean) == 0:
        print("No valid data to plot!")
    else:
        # Create plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Distribution of differences
        axes[0].hist(df_clean["difference"], bins=30, edgecolor="black", alpha=0.7)
        axes[0].axvline(0, color="red", linestyle="--", linewidth=2, label="Zero (no difference)")
        axes[0].axvline(df_clean["difference"].mean(), color="blue", linestyle="--", linewidth=2, 
                        label=f"Mean: {df_clean['difference'].mean():.4f}")
        axes[0].set_xlabel("Reconstruction Loss Difference\n(SCVI-WAE - SCVI)")
        axes[0].set_ylabel("Frequency")
        axes[0].set_title("Distribution of Reconstruction Loss Differences")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Scatter plot
        axes[1].scatter(df_clean["scvi_recon_loss"], df_clean["scvi_wae_recon_loss"], 
                       alpha=0.6, s=50)
        # Add diagonal line
        min_val = min(df_clean["scvi_recon_loss"].min(), df_clean["scvi_wae_recon_loss"].min())
        max_val = max(df_clean["scvi_recon_loss"].max(), df_clean["scvi_wae_recon_loss"].max())
        axes[1].plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="y=x")
        axes[1].set_xlabel("SCVI Reconstruction Loss")
        axes[1].set_ylabel("SCVI-WAE Reconstruction Loss")
        axes[1].set_title("SCVI vs SCVI-WAE Reconstruction Loss")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(args.output_plot, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {args.output_plot}")
        
        # Print summary statistics
        print(f"\nSummary Statistics:")
        print(f"  Number of paths: {len(df_clean)}")
        print(f"  Mean difference: {df_clean['difference'].mean():.4f}")
        print(f"  Median difference: {df_clean['difference'].median():.4f}")
        print(f"  Std difference: {df_clean['difference'].std():.4f}")
        print(f"  Min difference: {df_clean['difference'].min():.4f}")
        print(f"  Max difference: {df_clean['difference'].max():.4f}")
        print(f"\n  Paths where SCVI-WAE < SCVI (better): {(df_clean['difference'] < 0).sum()}")
        print(f"  Paths where SCVI-WAE > SCVI (worse): {(df_clean['difference'] > 0).sum()}")
