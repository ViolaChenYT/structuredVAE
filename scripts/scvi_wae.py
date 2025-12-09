import scanpy as sc
import argparse
import json
import gzip
import pickle
import os
import pandas as pd
import numpy as np
from typing import Dict
from scvi_wae.trainer import train_and_eval


def load_path_dict(path: str) -> Dict[str, dict]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        return json.load(f)
        
def load_json(path: str) -> Dict[str, dict]:
    with open(path, "r") as f:
        return json.load(f)

def parse_args():
    p = argparse.ArgumentParser(
        description="Train SCVI-WAE on a lineage path"
    )
    p.add_argument("--path_dict", type=str, \
        default = "/n/fs/ragr-data/users/viola/mouse_dev/scripts/new_c_elegans/paths_dict_min15.json.gz", \
        help="Input json for path dictionary")
    p.add_argument("--output_csv", type=str, \
        default = "/n/fs/ragr-data/users/viola/structuredVAE/results/scvi_wae_metrics.csv", \
        help="Output CSV file path for evaluation metrics")
    p.add_argument("--output_dir", type=str, \
        default = "/n/fs/ragr-data/users/viola/structuredVAE/results/scvi_wae_results", \
        help="Output directory for full results pickle files")
    return p.parse_args()

if __name__ == '__main__':
    # Load data
    args = parse_args()
    path_dict = load_path_dict(args.path_dict)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # List to store results
    results_list = []
    
    for path_name in path_dict:
        print(f"Training SCVI-WAE for {path_name}")
        try:
            adata = sc.read(f"/n/fs/ragr-data/users/viola/structuredVAE/data/scvi_path_{path_name}/trained.h5ad")
            results = train_and_eval(adata)
            
            # Prepare results dictionary for saving (convert tensors to numpy)
            results_to_save = {}
            for key, value in results.items():
                if key == "model" or key == "vae":
                    # Skip models - they're large and can be saved separately if needed
                    # For now, we'll save model state dicts
                    if key == "model":
                        results_to_save["model_state_dict"] = value.state_dict() if hasattr(value, "state_dict") else None
                    elif key == "vae":
                        # SCVI vae object - save state dict if available
                        if hasattr(value, "module") and hasattr(value.module, "state_dict"):
                            results_to_save["vae_state_dict"] = value.module.state_dict()
                        else:
                            results_to_save["vae_state_dict"] = None
                elif hasattr(value, "detach"):
                    # PyTorch tensor - convert to numpy
                    results_to_save[key] = value.detach().cpu().numpy()
                elif isinstance(value, (list, dict, np.ndarray)):
                    # Lists, dicts, numpy arrays - save as-is
                    results_to_save[key] = value
                else:
                    # Other types (scalars, strings, etc.)
                    results_to_save[key] = value
            
            # Save full results to pickle file
            output_pickle = os.path.join(args.output_dir, f"{path_name}_results.pkl")
            with open(output_pickle, "wb") as f:
                pickle.dump(results_to_save, f)
            print(f"  Saved full results to {output_pickle}")
            
            # Save losses_history as CSV for easier analysis
            if "losses_history" in results_to_save and results_to_save["losses_history"]:
                losses_df = pd.DataFrame(results_to_save["losses_history"])
                losses_csv = os.path.join(args.output_dir, f"{path_name}_losses.csv")
                losses_df.to_csv(losses_csv, index=False)
                print(f"  Saved losses history to {losses_csv}")
            
            # Extract evaluation metrics for CSV
            eval_results = results["eval_results"]
            metrics_row = {
                "path_name": path_name,
                "spearman_r": eval_results["spearman_r"],
                "spearman_p": eval_results["spearman_p"],
                "nmi_inbuilt": eval_results["nmi_inbuilt"],
                "ari_inbuilt": eval_results["ari_inbuilt"],
                "nmi_gmm": eval_results["nmi_gmm"],
                "ari_gmm": eval_results["ari_gmm"],
            }
            results_list.append(metrics_row)
            print(f"Trained SCVI-WAE for {path_name}")
        except Exception as e:
            print(f"Error training {path_name}: {e}")
            import traceback
            traceback.print_exc()
            # Still record the path with NaN values
            metrics_row = {
                "path_name": path_name,
                "spearman_r": None,
                "spearman_p": None,
                "nmi_inbuilt": None,
                "ari_inbuilt": None,
                "nmi_gmm": None,
                "ari_gmm": None,
            }
            results_list.append(metrics_row)
    
    # Create DataFrame and save to CSV
    df_results = pd.DataFrame(results_list)
    df_results.set_index("path_name", inplace=True)
    df_results.to_csv(args.output_csv)
    print(f"\nMetrics CSV saved to {args.output_csv}")
    print(f"Full results saved to {args.output_dir}/")
    print(f"\nSummary:")
    print(df_results.describe())
