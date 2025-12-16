"""
Plot latent variables vs cluster mixture proportions to assess prior flexibility.

This script helps determine whether the learnt prior is too flexible or not flexible enough
by visualizing:
1. Latent values vs posterior probabilities for each cluster
2. Learned mixture proportions
3. Distribution of latent values colored by cluster assignments
4. Comparison of learned vs uniform mixture proportions

Usage:
    # Plot a single result file:
    python scripts/plot_latent_vs_mixture.py --pickle_path results/scvi_wae_results_v5/path_name_results.pkl
    
    # Process all result files in a directory:
    python scripts/plot_latent_vs_mixture.py --batch --results_dir results/scvi_wae_results_v5

Interpretation:
    - KL divergence from uniform: 
      * < 0.1: Prior NOT FLEXIBLE ENOUGH (mixture proportions forced to uniform)
      * > 1.0: Prior TOO FLEXIBLE (some clusters dominate, others underused)
      * 0.1-1.0: Reasonable flexibility
    
    - Average max posterior probability:
      * < 0.7: Low confidence (uncertain cluster assignments)
      * > 0.95: High confidence (clear cluster membership)
      * 0.7-0.95: Moderate confidence (expected uncertainty at boundaries)
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import torch
from scipy.stats import entropy

# Import gmm_cluster_1d directly to avoid loading heavy dependencies
def gmm_cluster_1d(X, means, stds, mix_probs, eps=1e-8):
    """
    Cluster 1D data X using a Gaussian mixture model with *fixed* parameters,
    and compute log p(x_i) for each data point.
    """
    X = np.asarray(X, dtype=float)
    means = np.asarray(means, dtype=float)
    stds = np.asarray(stds, dtype=float)
    mix_probs = np.asarray(mix_probs, dtype=float)
    stds = np.maximum(stds, eps)
    mix_probs = mix_probs / mix_probs.sum()
    N = X.shape[0]
    K = means.shape[0]
    X_expanded = X[:, None]
    means_expanded = means[None]
    stds_expanded = stds[None]
    mix_expanded = mix_probs[None]
    var = stds_expanded ** 2
    log_norm_const = -0.5 * np.log(2.0 * np.pi * var)
    log_exp_term = -0.5 * (X_expanded - means_expanded) ** 2 / var
    log_comp = log_norm_const + log_exp_term
    log_joint = np.log(mix_expanded + eps) + log_comp
    log_joint_max = np.max(log_joint, axis=1, keepdims=True)
    joint_shifted = np.exp(log_joint - log_joint_max)
    sum_joint = np.sum(joint_shifted, axis=1, keepdims=True)
    log_px = (log_joint_max + np.log(sum_joint + eps)).ravel()
    responsibilities = joint_shifted / (sum_joint + eps)
    labels = np.argmax(responsibilities, axis=1)
    return labels, responsibilities, log_px


def load_results(pickle_path):
    """Load results from pickle file."""
    # Handle PyTorch tensors that may have been saved on CUDA
    # Monkey patch torch's restore_location to map CUDA to CPU
    old_restore = torch.serialization.default_restore_location
    
    def cpu_restore_location(storage, location):
        if isinstance(location, torch.device) and location.type == 'cuda':
            return storage.cpu()
        elif isinstance(location, str) and 'cuda' in location:
            return storage.cpu()
        else:
            return old_restore(storage, location)
    
    torch.serialization.default_restore_location = cpu_restore_location
    
    try:
        with open(pickle_path, "rb") as f:
            results = pickle.load(f)
    finally:
        # Restore original function
        torch.serialization.default_restore_location = old_restore
    
    # Convert any remaining torch tensors to numpy
    for key, value in results.items():
        if isinstance(value, torch.Tensor):
            results[key] = value.detach().cpu().numpy()
        elif hasattr(value, 'detach') and hasattr(value, 'cpu'):
            try:
                results[key] = value.detach().cpu().numpy()
            except:
                pass  # Skip if conversion fails
    
    return results


def compute_mixture_probs(mix_logits):
    """Convert mixture logits to probabilities."""
    if isinstance(mix_logits, np.ndarray):
        mix_logits_tensor = mix_logits
    else:
        # If it's a torch tensor, convert to numpy
        mix_logits_tensor = mix_logits.detach().cpu().numpy() if hasattr(mix_logits, 'detach') else np.array(mix_logits)
    
    # Softmax to get probabilities
    exp_logits = np.exp(mix_logits_tensor - np.max(mix_logits_tensor))
    probs = exp_logits / exp_logits.sum()
    return probs


def plot_latent_vs_mixture(results, output_path=None, path_name=None):
    """
    Plot latent variables vs cluster mixture proportions.
    
    Parameters
    ----------
    results : dict
        Dictionary containing:
        - 'mu': latent representations (N,)
        - 'mix_logits': learned mixture logits (K,)
        - 'centroids': learned centroids (K,)
        - 'log_stds': learned log standard deviations (K,)
    output_path : str, optional
        Path to save the plot
    path_name : str, optional
        Name of the path for title
    """
    # Extract data
    mu = results['mu']  # (N,)
    mix_logits = results['mix_logits']  # (K,)
    centroids = results['centroids']  # (K,)
    log_stds = results['log_stds']  # (K,)
    
    # Convert to numpy if needed
    if hasattr(mix_logits, 'detach'):
        mix_logits = mix_logits.detach().cpu().numpy()
    if hasattr(centroids, 'detach'):
        centroids = centroids.detach().cpu().numpy()
    if hasattr(log_stds, 'detach'):
        log_stds = log_stds.detach().cpu().numpy()
    
    # Compute mixture proportions
    mix_probs = compute_mixture_probs(mix_logits)
    n_clusters = len(mix_probs)
    
    # Compute standard deviations
    stds = np.exp(log_stds)
    
    # Compute posterior probabilities (responsibilities) for each data point
    labels, responsibilities, log_px = gmm_cluster_1d(
        mu.flatten(),
        centroids.flatten(),
        stds.flatten(),
        mix_probs
    )
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Latent values vs posterior probabilities for each cluster
    ax1 = plt.subplot(2, 3, 1)
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    for k in range(n_clusters):
        ax1.scatter(mu, responsibilities[:, k], alpha=0.3, s=10, 
                   label=f'Cluster {k+1}', color=colors[k])
    ax1.set_xlabel('Latent Value (μ)')
    ax1.set_ylabel('Posterior Probability')
    ax1.set_title('Latent vs Posterior Probabilities')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Distribution of latent values colored by cluster assignments
    ax2 = plt.subplot(2, 3, 2)
    for k in range(n_clusters):
        mask = labels == k
        ax2.hist(mu[mask], bins=50, alpha=0.5, label=f'Cluster {k+1}', 
                color=colors[k], density=True)
    # Overlay learned prior components
    x_range = np.linspace(mu.min(), mu.max(), 200)
    for k in range(n_clusters):
        prior_density = mix_probs[k] * (1 / (stds[k] * np.sqrt(2 * np.pi))) * \
                       np.exp(-0.5 * ((x_range - centroids[k]) / stds[k]) ** 2)
        ax2.plot(x_range, prior_density, '--', linewidth=2, 
                color=colors[k], label=f'Prior {k+1}' if k == 0 else '')
    ax2.set_xlabel('Latent Value (μ)')
    ax2.set_ylabel('Density')
    ax2.set_title('Latent Distribution vs Prior Components')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Learned mixture proportions (bar plot)
    ax3 = plt.subplot(2, 3, 3)
    x_pos = np.arange(n_clusters)
    uniform_probs = np.ones(n_clusters) / n_clusters
    width = 0.35
    ax3.bar(x_pos - width/2, mix_probs, width, label='Learned', alpha=0.7, color='steelblue')
    ax3.bar(x_pos + width/2, uniform_probs, width, label='Uniform', alpha=0.7, color='lightcoral')
    ax3.set_xlabel('Cluster')
    ax3.set_ylabel('Mixture Proportion')
    ax3.set_title('Learned vs Uniform Mixture Proportions')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f'C{i+1}' for i in range(n_clusters)])
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Entropy of posterior probabilities (uncertainty)
    ax4 = plt.subplot(2, 3, 4)
    entropies = [entropy(responsibilities[i]) for i in range(len(responsibilities))]
    ax4.scatter(mu, entropies, alpha=0.3, s=10, c=labels, cmap='tab10')
    ax4.set_xlabel('Latent Value (μ)')
    ax4.set_ylabel('Entropy of Posterior')
    ax4.set_title('Uncertainty in Cluster Assignment')
    ax4.grid(True, alpha=0.3)
    cbar = plt.colorbar(ax4.collections[0], ax=ax4)
    cbar.set_label('Cluster Assignment')
    
    # 5. Maximum posterior probability (confidence)
    ax5 = plt.subplot(2, 3, 5)
    max_probs = np.max(responsibilities, axis=1)
    ax5.scatter(mu, max_probs, alpha=0.3, s=10, c=labels, cmap='tab10')
    ax5.set_xlabel('Latent Value (μ)')
    ax5.set_ylabel('Max Posterior Probability')
    ax5.set_title('Confidence in Cluster Assignment')
    ax5.grid(True, alpha=0.3)
    cbar = plt.colorbar(ax5.collections[0], ax=ax5)
    cbar.set_label('Cluster Assignment')
    
    # 6. Prior components visualization
    ax6 = plt.subplot(2, 3, 6)
    x_range = np.linspace(mu.min() - 2*stds.max(), mu.max() + 2*stds.max(), 300)
    for k in range(n_clusters):
        prior_density = mix_probs[k] * (1 / (stds[k] * np.sqrt(2 * np.pi))) * \
                       np.exp(-0.5 * ((x_range - centroids[k]) / stds[k]) ** 2)
        ax6.plot(x_range, prior_density, linewidth=2, label=f'Cluster {k+1}', color=colors[k])
        ax6.axvline(centroids[k], linestyle='--', color=colors[k], alpha=0.5)
    # Overall mixture
    overall_density = np.zeros_like(x_range)
    for k in range(n_clusters):
        overall_density += mix_probs[k] * (1 / (stds[k] * np.sqrt(2 * np.pi))) * \
                          np.exp(-0.5 * ((x_range - centroids[k]) / stds[k]) ** 2)
    ax6.plot(x_range, overall_density, 'k-', linewidth=2, label='Mixture', alpha=0.7)
    ax6.hist(mu, bins=50, density=True, alpha=0.3, color='gray', label='Data')
    ax6.set_xlabel('Latent Value (μ)')
    ax6.set_ylabel('Density')
    ax6.set_title('Learned Prior Components')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Add overall title
    title = f'Latent vs Mixture Analysis'
    if path_name:
        title += f' - {path_name}'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    else:
        plt.show()
    
    plt.close()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Number of clusters: {n_clusters}")
    print(f"\nLearned mixture proportions:")
    for k in range(n_clusters):
        print(f"  Cluster {k+1}: {mix_probs[k]:.4f}")
    print(f"\nUniform mixture proportions:")
    for k in range(n_clusters):
        print(f"  Cluster {k+1}: {uniform_probs[k]:.4f}")
    
    # KL divergence from uniform
    kl_from_uniform = entropy(mix_probs, uniform_probs)
    print(f"\nKL divergence from uniform: {kl_from_uniform:.4f}")
    print(f"  (Higher = more flexible, Lower = closer to uniform)")
    
    # Average entropy of posterior
    avg_entropy = np.mean(entropies)
    print(f"\nAverage entropy of posterior: {avg_entropy:.4f}")
    print(f"  (Higher = more uncertain, Lower = more confident)")
    
    # Average max probability
    avg_max_prob = np.mean(max_probs)
    print(f"\nAverage max posterior probability: {avg_max_prob:.4f}")
    print(f"  (Higher = more confident, Lower = more uncertain)")
    
    # Interpretation
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    if kl_from_uniform < 0.1:
        print("⚠️  Prior appears NOT FLEXIBLE ENOUGH:")
        print("   - Mixture proportions are very close to uniform")
        print("   - The model may be forcing equal cluster sizes")
    elif kl_from_uniform > 1.0:
        print("⚠️  Prior appears TOO FLEXIBLE:")
        print("   - Mixture proportions deviate significantly from uniform")
        print("   - Some clusters may dominate while others are underused")
    else:
        print("✓ Prior flexibility appears reasonable:")
        print("   - Mixture proportions show moderate deviation from uniform")
        print("   - This suggests the model can adapt to data structure")
    
    if avg_max_prob < 0.7:
        print("\n⚠️  Low confidence in cluster assignments:")
        print("   - Many data points have uncertain cluster membership")
        print("   - Consider if clusters are well-separated")
    elif avg_max_prob > 0.95:
        print("\n✓ High confidence in cluster assignments:")
        print("   - Most data points have clear cluster membership")
    else:
        print("\n✓ Moderate confidence in cluster assignments:")
        print("   - Some uncertainty is expected in boundary regions")


def main():
    parser = argparse.ArgumentParser(
        description="Plot latent variables vs cluster mixture proportions"
    )
    parser.add_argument(
        "--pickle_path",
        type=str,
        help="Path to results pickle file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save output plot (default: same as pickle with .png extension)"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="/n/fs/ragr-data/users/viola/structuredVAE/results/scvi_wae_results_v5",
        help="Directory containing result pickle files (for batch processing)"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process all pickle files in results_dir"
    )
    
    args = parser.parse_args()
    
    if args.batch:
        # Process all pickle files in directory
        import glob
        pickle_files = glob.glob(os.path.join(args.results_dir, "*_results.pkl"))
        print(f"Found {len(pickle_files)} result files")
        
        for pickle_path in pickle_files:
            path_name = os.path.basename(pickle_path).replace("_results.pkl", "")
            output_path = pickle_path.replace(".pkl", "_latent_vs_mixture.png")
            
            print(f"\nProcessing {path_name}...")
            try:
                results = load_results(pickle_path)
                plot_latent_vs_mixture(results, output_path=output_path, path_name=path_name)
            except Exception as e:
                print(f"Error processing {path_name}: {e}")
                import traceback
                traceback.print_exc()
    else:
        # Process single file
        results = load_results(args.pickle_path)
        
        if args.output_path is None:
            args.output_path = args.pickle_path.replace(".pkl", "_latent_vs_mixture.png")
        
        path_name = os.path.basename(args.pickle_path).replace("_results.pkl", "")
        plot_latent_vs_mixture(results, output_path=args.output_path, path_name=path_name)


if __name__ == "__main__":
    main()
