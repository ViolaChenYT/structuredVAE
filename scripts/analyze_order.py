import anndata
import argparse
from scipy.stats import spearmanr
import pandas as pd

def analyze_correlation(
    adata_path: str,
    obs_key: str = 'lineage_node',
    obsm_key: str = 'X_scVI_1d'
) -> None:
    """
    Calculates the Spearman rank correlation between an ordinal categorical variable
    in .obs and a 1D continuous variable in .obsm.

    This statistic, presented as its absolute value, measures how well the 1D
    embedding captures the ordering of the categorical lineage nodes. A value
    close to 1 indicates a strong monotonic relationship (either increasing or
    decreasing), while a value close to 0 indicates no relationship.

    Args:
        adata_path (str): Path to the .h5ad AnnData file.
        obs_key (str): The key in adata.obs containing the ordinal categories (e.g., 'lineage_node').
        obsm_key (str): The key in adata.obsm for the 1D embedding (e.g., 'X_scVI_1d').
    """
    try:
        # --- 1. Load Data ---
        print(f"Loading data from {adata_path}...")
        adata = anndata.read_h5ad(adata_path)
        print("Data loaded successfully.")

        # --- 2. Extract Data Series ---
        # Note: We assume the AnnData object is already filtered for the single
        # path/lineage you wish to analyze.
        if obs_key not in adata.obs.columns:
            raise KeyError(f"Error: Column '{obs_key}' not found in adata.obs.")
        if obsm_key not in adata.obsm:
            raise KeyError(f"Error: Key '{obsm_key}' not found in adata.obsm.")

        lineage_nodes = adata.obs[obs_key]
        embedding_1d = adata.obsm[obsm_key].flatten()

        # Ensure we're working with numeric types for correlation
        # The categories might be stored as strings ('1', '2'), so convert them.
        try:
            lineage_nodes_numeric = pd.to_numeric(lineage_nodes)
        except ValueError:
            raise TypeError(
                f"Column '{obs_key}' contains non-numeric values and could not be coerced. "
                "Please ensure it contains only orderable numbers (e.g., 1, 2, 3...)."
            )

        print(f"Found {len(lineage_nodes_numeric)} cells to analyze.")
        print(f"Analyzing correlation between '{obs_key}' and '{obsm_key}'.")

        # --- 3. Calculate Correlation ---
        correlation, p_value = spearmanr(lineage_nodes_numeric, embedding_1d)

        # --- 4. Report Results ---
        print("\n--- Results ---")
        print(f"Spearman's rho (ρ): {correlation:.4f}")
        print(f"p-value: {p_value:.4e}")
        print("-----------------")
        print(f"Statistic |ρ|: {abs(correlation):.4f}")
        print("-----------------")
        print("\nInterpretation:")
        print("The statistic |ρ| measures the strength of the monotonic relationship.")
        print(" - A value near 1.0 indicates the 1D embedding strongly captures the lineage order.")
        print(" - A value near 0.0 indicates the 1D embedding does not capture the lineage order.")

    except FileNotFoundError:
        print(f"Error: The file '{adata_path}' was not found.")
    except (KeyError, TypeError) as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate Spearman correlation for lineage ordering in an AnnData object.")
    parser.add_argument("adata_path",type=str, required=True, help="Path to the input AnnData file (.h5ad).")
    parser.add_argument("--obs_key",type=str,default="lineage",
        help="Key in adata.obs for the categorical lineage nodes. Default: 'lineage_node'.")
    parser.add_argument("--obsm_key",type=str,default="X_scVI",
        help="Key in adata.obsm for the 1D embedding. Default: 'X_scVI'.")
    args = parser.parse_args()
    analyze_correlation(args.adata_path, args.obs_key, args.obsm_key)