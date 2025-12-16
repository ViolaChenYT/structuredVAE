import os
import tempfile
import time
from pathlib import Path
from typing import Optional, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import plotnine as p9
import scanpy as sc
import scvi
import seaborn as sns
import torch
import anndata
import networkx as nx
import sys
import argparse

def parse_args():
    p = argparse.ArgumentParser(
        description="Train scVI on a lineage path subset between start and end prefixes."
    )
    p.add_argument("start", type=str, help="Prefix lineage label (e.g., MSxap)")
    p.add_argument("end", type=str, help="Full lineage label ending with the path (e.g., MSxappppx)")
    p.add_argument("--remake-plots", action="store_true", 
                   help="If set, remake plots from existing trained.h5ad files. If trained.h5ad doesn't exist, train normally.")
    p.add_argument("--remake-reconstruction-plots-only", action="store_true",
                   help="If set, only remake reconstruction loss plots from existing training_history.csv. Skips all other plots and training.")
    #p.add_argument("--adata", type=Path, default=Path(
    #    "/n/fs/ragr-data/users/yihangs/Celegan/structuredVAE/data/packer2019_preprocessed.h5ad"
    #), help="Input AnnData .h5ad file")
    #p.add_argument("--outdir-root", type=Path, default=Path(
    #    "/n/fs/ragr-data/users/yihangs/Celegan/structuredVAE/data"
    #), help="Root directory for outputs")
    #p.add_argument("--max-epochs", type=int, default=1000, help="Max epochs for scVI training")
    #p.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    #p.add_argument("--device", type=int, default=0, help="CUDA device index for training")
    return p.parse_args()


def train_scvi_on_path(
    adata: anndata.AnnData,
    lineage_path: list,
    path_name: str,
    base_path: str = "/n/fs/ragr-data/users/viola/structuredVAE/",
    save_loom: bool = False,
    save_model: bool = True,
    save_plots: bool = False,
    arches_params: Optional[Dict[str, Any]] = None,
    training_params: Optional[Dict[str, Any]] = None,
    group_to_nodes: Optional[Dict[str, list]] = None,
    fuzzy_mapping: Optional[Dict[str, list]] = None,
    remake_plots_only: bool = False,
    remake_reconstruction_plots_only: bool = False
) -> Dict[str, Any]:
    """
    Train scVI model on a lineage path subset.
    
    Parameters:
    -----------
    adata : anndata.AnnData
        Input AnnData object containing the full dataset
    lineage_path : list
        List of lineage labels representing the complete path (e.g., ["MSxap", "MSxapp", "MSxappp", "MSxappppx"])
    path_name : str
        Name for the path (used for file naming)
    base_path : str, optional
        Base path for saving outputs (default: "/n/fs/ragr-data/users/viola/structuredVAE/")
    save_loom : bool, optional
        Whether to save the filtered data as loom file (default: True)
    save_model : bool, optional
        Whether to save the trained model (default: True)
    save_plots : bool, optional
        Whether to save training plots (default: True)
    arches_params : dict, optional
        Architecture parameters for scVI model (default: None, uses default params)
    training_params : dict, optional
        Training parameters for scVI model (default: None, uses default params)
    group_to_nodes : dict, optional
        Dictionary mapping group names to lists of nodes (default: None)
    fuzzy_mapping : dict, optional
        Dictionary mapping nodes to lists of possible lineage values in adata.obs.lineage (default: None)
    remake_plots_only : bool, optional
        If True, check if trained.h5ad exists and only remake plots without retraining (default: False)
    
    Returns:
    --------
    dict
        Dictionary containing:
        - 'adata_path': Filtered AnnData object
        - 'vae': Trained scVI model
        - 'model_path': Path where model was saved
        - 'training_history': Training history DataFrame
    """
    # Validate inputs
    assert len(lineage_path) > 0, "Lineage path cannot be empty"
    assert all(isinstance(label, str) for label in lineage_path), "All lineage labels must be strings"
    
    # Get model path
    model_path = f"{base_path}/data/scvi_path_{path_name}/"
    trained_h5ad_path = os.path.join(model_path, 'trained.h5ad')
    training_history_path = os.path.join(model_path, 'training_history.csv')
    
    # If remake_reconstruction_plots_only is True, only plot reconstruction loss
    if remake_reconstruction_plots_only:
        if not os.path.exists(training_history_path):
            print(f"  Warning: {training_history_path} not found, skipping {path_name}")
            print(f"  Cannot generate reconstruction loss plot without training history.")
            return {
                'adata_path': None,
                'vae': None,
                'model_path': model_path,
                'training_history': pd.DataFrame()
            }
        
        print(f"Remaking reconstruction loss plots for {path_name}...")
        train_test_results = pd.read_csv(training_history_path)
        
        # Check for reconstruction loss columns
        if "reconstruction_loss_train" in train_test_results.columns:
            recon_loss_train = train_test_results["reconstruction_loss_train"].values
        elif "reconstruction_error_train" in train_test_results.columns:
            recon_loss_train = train_test_results["reconstruction_error_train"].values
        else:
            recon_loss_train = None
        
        if "reconstruction_loss_validation" in train_test_results.columns:
            recon_loss_validation = train_test_results["reconstruction_loss_validation"].values
        elif "reconstruction_error_validation" in train_test_results.columns:
            recon_loss_validation = train_test_results["reconstruction_error_validation"].values
        else:
            recon_loss_validation = None
        
        # Plot reconstruction loss if available
        if recon_loss_train is not None or recon_loss_validation is not None:
            plt.figure(figsize=(8, 6))
            epochs = np.arange(len(train_test_results))
            if recon_loss_train is not None:
                plt.plot(epochs, recon_loss_train, label='Train Reconstruction Loss', alpha=0.7)
            if recon_loss_validation is not None:
                plt.plot(epochs, recon_loss_validation, label='Validation Reconstruction Loss', alpha=0.7)
            plt.xlabel("Epoch")
            plt.ylabel("Reconstruction Loss")
            plt.title(f"Reconstruction Loss: {path_name}")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(model_path, "reconstruction_loss.png"), dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✓ Successfully remade reconstruction loss plot for {path_name}")
        else:
            print(f"  Warning: No reconstruction loss data found in {training_history_path}")
            print(f"  Available columns: {list(train_test_results.columns)}")
            print(f"  Note: Reconstruction loss may not have been saved during training.")
        
        return {
            'adata_path': None,
            'vae': None,
            'model_path': model_path,
            'training_history': train_test_results
        }
    
    # If remake_plots_only is True and trained.h5ad exists, load and remake plots
    if remake_plots_only and os.path.exists(trained_h5ad_path):
        print(f"Remaking plots for {path_name} from existing trained.h5ad...")
        adata_path = sc.read(trained_h5ad_path)
        
        # Load the trained model
        vae = scvi.model.SCVI.load(model_path, adata=adata_path)
        
        # Load training history if it exists
        training_history_path = os.path.join(model_path, 'training_history.csv')
        if os.path.exists(training_history_path):
            train_test_results = pd.read_csv(training_history_path)
            # Check for reconstruction loss columns
            if "reconstruction_loss_train" in train_test_results.columns:
                recon_loss_train = train_test_results["reconstruction_loss_train"].values
            else:
                recon_loss_train = None
            if "reconstruction_loss_validation" in train_test_results.columns:
                recon_loss_validation = train_test_results["reconstruction_loss_validation"].values
            else:
                recon_loss_validation = None
        else:
            train_test_results = pd.DataFrame()
            recon_loss_train = None
            recon_loss_validation = None
        
        # Generate plots if requested
        if save_plots:
            # Plot reconstruction loss if available
            if recon_loss_train is not None or recon_loss_validation is not None:
                plt.figure(figsize=(8, 6))
                epochs = np.arange(len(train_test_results))
                if recon_loss_train is not None:
                    plt.plot(epochs, recon_loss_train, label='Train Reconstruction Loss', alpha=0.7)
                if recon_loss_validation is not None:
                    plt.plot(epochs, recon_loss_validation, label='Validation Reconstruction Loss', alpha=0.7)
                plt.xlabel("Epoch")
                plt.ylabel("Reconstruction Loss")
                plt.title(f"Reconstruction Loss: {path_name}")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(model_path, "reconstruction_loss.png"), dpi=150, bbox_inches='tight')
                plt.close()
            
            # 1D latent distribution
            arr = adata_path.obsm["X_scVI"].flatten()
            plt.hist(arr, bins=50, edgecolor='black')
            plt.xlabel("Value")
            plt.ylabel("Frequency")
            plt.title("Frequency Distribution")
            plt.savefig(model_path + "1dlatent_frequency_distribution.png")
            plt.close()
            
            # Distribution by lineage
            labels = adata_path.obs["lineage"].tolist()
            unique_labels = np.unique(labels)
            unique_labels = sorted(unique_labels, key=lambda x: len(str(x)))
            
            # Get all coordinates to determine global range for shared binning
            global_min = np.min(arr)
            global_max = np.max(arr)
            # Create shared bin edges (201 edges = 200 bins)
            # Using more bins since we're now binning across all lineages, not per lineage
            bin_edges = np.linspace(global_min, global_max, 201)
            
            plt.figure(figsize=(7, 5))
            for lab in unique_labels:
                subset = arr[np.array(labels) == lab]
                # Add cell count to legend label
                n_cells = len(subset)
                label = f"{lab} (n={n_cells})"
                plt.hist(subset, bins=bin_edges, density=True, alpha=0.5, label=label)
            
            plt.xlabel("Value")
            plt.ylabel("Density")
            plt.title("Frequency Distributions by Label (normalized)")
            plt.legend()
            plt.savefig(model_path + "1dlatent_frequency_distribution_by_lineage.png")
            plt.close()
        
        print(f"✓ Successfully remade plots for {path_name}")
        
        return {
            'adata_path': adata_path,
            'vae': vae,
            'model_path': model_path,
            'training_history': train_test_results
        }
    
    # Expand lineage path using fuzzy mapping if provided
    expanded_lineage_path = lineage_path.copy()
    
    if fuzzy_mapping is not None:
        print("Using fuzzy mapping to expand lineage path...")
        expanded_lineage_set = set()
        for lineage_label in lineage_path:
            if group_to_nodes is not None:
                # First, look up in group_to_nodes to find corresponding nodes
                if lineage_label in group_to_nodes:
                    nodes = group_to_nodes[lineage_label]
                    print(f"  {lineage_label} -> nodes: {nodes}")
            else:
                nodes = [lineage_label]
                
                # For each node, look up in fuzzy_mapping to find possible lineage values
            for node in nodes:
                if node in fuzzy_mapping:
                    possible_lineages = fuzzy_mapping[node]
                    print(f"    {node} -> possible lineages: {possible_lineages}")
                    expanded_lineage_set.update(possible_lineages)
                else:
                    print(f"    Warning: Node {node} not found in fuzzy_mapping")
                    expanded_lineage_set.add(node)
        
        expanded_lineage_path = list(expanded_lineage_set)
        print(f"Expanded lineage path: {lineage_path} -> {expanded_lineage_path}")
    else: print("No fuzzy mapping provided, using original lineage path")
    # Default architecture parameters
    if arches_params is None:
        arches_params = dict(
            use_layer_norm="both",
            use_batch_norm="none",
            encode_covariates=True,
            dropout_rate=0.2,
            n_layers=2,
            n_hidden=64,
            n_latent=1,
        )
    
    # Default training parameters
    if training_params is None:
        training_params = dict(
            check_val_every_n_epoch=1,
            max_epochs=1000,
            early_stopping=True,
            early_stopping_patience=20,
            early_stopping_monitor="elbo_validation",
            # devices=[0],
            accelerator="cpu",
        )
    
    # Filter data to expanded lineage path
    adata_path = adata[adata.obs["lineage"].isin(expanded_lineage_path)].copy()
    
    # Validate that lineages are present
    found_lineages = set(adata_path.obs["lineage"].unique().tolist())
    expected_lineages = set(expanded_lineage_path)
    missing_lineages = expected_lineages - found_lineages
    
    if missing_lineages:
        print(f"Warning: Missing lineages in data: {missing_lineages}")
        # Filter to only include lineages that exist in the data
        expanded_lineage_path = [label for label in expanded_lineage_path if label in found_lineages]
        adata_path = adata[adata.obs["lineage"].isin(expanded_lineage_path)].copy()
    
    # Save loom file if requested
    if save_loom:
        loom_path = f"{base_path}/data/{path_name}.loom"
        adata_path.write_loom(loom_path)
    
    # Setup scVI
    scvi.model.SCVI.setup_anndata(adata_path, layer="counts", batch_key="batch")
    
    # Create and train model
    vae = scvi.model.SCVI(adata_path, **arches_params)
    vae.train(**training_params)
    
    # Save model if requested
    if save_model:
        vae.save(model_path, overwrite=True)
    
    # Get latent representation and save processed data
    adata_path.obsm["X_scVI"] = vae.get_latent_representation()
    
    # Compute 50-dimensional PCA
    sc.tl.pca(adata_path, n_comps=50, use_highly_variable=False)
    # Compute 1D UMAP from 50-dimensional PCA
    sc.pp.neighbors(adata_path, n_neighbors=15, use_rep='X_pca')
    sc.tl.umap(adata_path, n_components=1)

    sc.tl.pca(adata_path, n_comps=1, use_highly_variable=False)
    
    if save_model:
        adata_path.write_h5ad(model_path + '/trained.h5ad')
    
    # Prepare training history
    train_test_results = vae.history["elbo_train"].copy()
    train_test_results["elbo_validation"] = vae.history["elbo_validation"]
    
    # Check for reconstruction loss in history
    recon_loss_train = None
    recon_loss_validation = None
    if "reconstruction_loss_train" in vae.history:
        recon_loss_train = vae.history["reconstruction_loss_train"]
        train_test_results["reconstruction_loss_train"] = recon_loss_train
    elif "reconstruction_error_train" in vae.history:
        recon_loss_train = vae.history["reconstruction_error_train"]
        train_test_results["reconstruction_loss_train"] = recon_loss_train
    
    if "reconstruction_loss_validation" in vae.history:
        recon_loss_validation = vae.history["reconstruction_loss_validation"]
        train_test_results["reconstruction_loss_validation"] = recon_loss_validation
    elif "reconstruction_error_validation" in vae.history:
        recon_loss_validation = vae.history["reconstruction_error_validation"]
        train_test_results["reconstruction_loss_validation"] = recon_loss_validation
    
    if save_model:
        train_test_results.to_csv(model_path + '/training_history.csv', index=False)
    
    # Generate plots if requested
    if save_plots:
        # Plot reconstruction loss if available
        if recon_loss_train is not None or recon_loss_validation is not None:
            plt.figure(figsize=(8, 6))
            epochs = np.arange(len(train_test_results))
            if recon_loss_train is not None:
                plt.plot(epochs, recon_loss_train, label='Train Reconstruction Loss', alpha=0.7)
            if recon_loss_validation is not None:
                plt.plot(epochs, recon_loss_validation, label='Validation Reconstruction Loss', alpha=0.7)
            plt.xlabel("Epoch")
            plt.ylabel("Reconstruction Loss")
            plt.title(f"Reconstruction Loss: {path_name}")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(model_path, "reconstruction_loss.png"), dpi=150, bbox_inches='tight')
            plt.close()
        
        # Distribution by lineage
        arr = adata_path.obsm["X_scVI"].flatten()
        labels = adata_path.obs["lineage"].tolist()
        unique_labels = np.unique(labels)
        unique_labels = sorted(unique_labels, key=lambda x: len(str(x)))
        
        # Get all coordinates to determine global range for shared binning
        global_min = np.min(arr)
        global_max = np.max(arr)
        # Create shared bin edges (201 edges = 200 bins)
        # Using more bins since we're now binning across all lineages, not per lineage
        bin_edges = np.linspace(global_min, global_max, 201)
        
        plt.figure(figsize=(7, 5))
        for lab in unique_labels:
            subset = arr[np.array(labels) == lab]
            # Add cell count to legend label
            n_cells = len(subset)
            label = f"{lab} (n={n_cells})"
            plt.hist(subset, bins=bin_edges, density=True, alpha=0.5, label=label)
        
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.title("Frequency Distributions by Label (normalized)")
        plt.legend()
        plt.savefig(model_path + "1dlatent_frequency_distribution_by_lineage.png")
        plt.close()
    
    return {
        'adata_path': adata_path,
        'vae': vae,
        'model_path': model_path,
        'training_history': train_test_results
    }


def main():
    start_time = time.time()
    
    base_path = "/n/fs/ragr-data/users/viola/structuredVAE/"
    adata = sc.read(f"{base_path}/data/packer2019_preprocessed.h5ad")

    args = parse_args()
    start = args.start
    end = args.end
    
    # Generate lineage path from start and end for backward compatibility
    lineage_path = [end[:len(start)+i] for i in range(0, len(end)-len(start)+1)]
    
    # Use the modular function
    result = train_scvi_on_path(
        adata=adata,
        lineage_path=lineage_path,
        path_name=f"{start}_to_{end}",
        base_path=base_path,
        remake_plots_only=args.remake_plots,
        remake_reconstruction_plots_only=args.remake_reconstruction_plots_only
    )
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"Training completed successfully!")
    print(f"Model saved to: {result['model_path']}")
    print(f"Filtered data shape: {result['adata_path'].shape}")
    print(f"Training epochs: {len(result['training_history'])}")
    print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")


if __name__ == "__main__":
    main()