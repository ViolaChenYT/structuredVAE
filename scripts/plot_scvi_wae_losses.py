"""
Visualize SCVI-WAE training losses over epochs for a specific path.

This script loads loss history from CSV files and creates a plot showing
how different loss terms change during training.
"""
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Plot SCVI-WAE training losses for a specific path"
    )
    p.add_argument(
        "path_name",
        type=str,
        help="Name of the path (e.g., MSppa_MSppaaaav)"
    )
    p.add_argument(
        "--input_dir",
        type=str,
        default="/n/fs/ragr-data/users/viola/structuredVAE/results/scvi_wae_results_v5",
        help="Directory containing loss CSV files (default: results/scvi_wae_results)"
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output PNG file path (default: {input_dir}/{path_name}_losses.png)"
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for output image (default: 150)"
    )
    return p.parse_args()


def plot_losses(losses_df, path_name, output_path, dpi=150):
    """
    Plot all loss terms over training epochs.
    
    Parameters
    ----------
    losses_df : pd.DataFrame
        DataFrame containing loss history with columns:
        - ae_loss, ae_recon_loss, ae_wd, ae_dist
        - prior_loss, prior_wd (optional)
    path_name : str
        Name of the path (for title)
    output_path : str
        Path to save the plot
    dpi : int
        DPI for the output image
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define colors and styles for each loss term
    loss_configs = {
        "ae_loss": {"color": "#1f77b4", "linestyle": "-", "linewidth": 2, "label": "AE Total Loss"},
        "ae_recon_loss": {"color": "#ff7f0e", "linestyle": "-", "linewidth": 1.5, "label": "AE Reconstruction Loss"},
        "ae_wd": {"color": "#2ca02c", "linestyle": "--", "linewidth": 1.5, "label": "AE Wasserstein Distance"},
        "ae_dist": {"color": "#d62728", "linestyle": "--", "linewidth": 1.5, "label": "AE Pairwise Distance"},
        "prior_loss": {"color": "#9467bd", "linestyle": "-", "linewidth": 2, "label": "Prior Total Loss"},
        "prior_wd": {"color": "#8c564b", "linestyle": "-.", "linewidth": 1.5, "label": "Prior Wasserstein Distance"},
    }
    
    # Get epoch numbers (1-indexed)
    epochs = np.arange(1, len(losses_df) + 1)
    
    # Plot each loss term that exists in the dataframe
    for loss_name, config in loss_configs.items():
        if loss_name in losses_df.columns:
            values = losses_df[loss_name].values
            if loss_name == "ae_wd":
                values = values * 100
            ax.plot(
                epochs,
                values,
                color=config["color"],
                linestyle=config["linestyle"],
                linewidth=config["linewidth"],
                label=config["label"],
                alpha=0.8
            )
    
    # Style the plot
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss Value", fontsize=12)
    ax.set_title(f"Training Losses: {path_name}", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")
    
    # Set x-axis to show integer epochs
    ax.set_xticks(np.arange(0, len(epochs), max(1, len(epochs) // 10)))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    
    print(f"Plot saved to {output_path}")


def main():
    """Main function."""
    args = parse_args()
    
    # Construct input CSV path
    input_csv = os.path.join(args.input_dir, f"{args.path_name}_losses.csv")
    
    # Check if file exists
    if not os.path.exists(input_csv):
        raise FileNotFoundError(
            f"Loss CSV file not found: {input_csv}\n"
            f"Please check that the path name is correct and the file exists."
        )
    
    # Load CSV file
    print(f"Loading losses from {input_csv}...")
    try:
        losses_df = pd.read_csv(input_csv)
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")
    
    # Check if dataframe is empty
    if len(losses_df) == 0:
        raise ValueError(f"CSV file is empty: {input_csv}")
    
    # Print available columns
    print(f"Found {len(losses_df)} epochs")
    print(f"Available loss columns: {list(losses_df.columns)}")
    
    # Determine output path
    if args.output is None:
        output_path = os.path.join(args.input_dir, f"{args.path_name}_losses.png")
    else:
        output_path = args.output
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Plot losses
    plot_losses(losses_df, args.path_name, output_path, dpi=args.dpi)
    
    # Print summary statistics
    print("\nLoss Summary Statistics:")
    for col in losses_df.columns:
        if losses_df[col].dtype in [np.float64, np.int64]:
            print(f"  {col}:")
            print(f"    Initial: {losses_df[col].iloc[0]:.4f}")
            print(f"    Final: {losses_df[col].iloc[-1]:.4f}")
            print(f"    Min: {losses_df[col].min():.4f}")
            print(f"    Max: {losses_df[col].max():.4f}")


if __name__ == "__main__":
    main()
