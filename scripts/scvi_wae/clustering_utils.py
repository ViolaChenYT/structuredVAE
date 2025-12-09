import numpy as np


def gmm_cluster_1d(X, means, stds, mix_probs, eps=1e-8):
    """
    Cluster 1D data X using a Gaussian mixture model with *fixed* parameters,
    and compute log p(x_i) for each data point.
    Parameters
    ----------
    X : array-like, shape (N,)
        Data points.
    means : array-like, shape (K,)
        Means of each Gaussian component.
    stds : array-like, shape (K,)
        Standard deviations of each Gaussian component.
    mix_probs : array-like, shape (K,)
        Mixture weights (should sum to 1).
    Returns
    -------
    labels : ndarray, shape (N,)
        Hard cluster assignments (0..K-1).
    responsibilities : ndarray, shape (N, K)
        Posterior probabilities p(z=k | x_i).
    log_px : ndarray, shape (N,)
        Log-likelihood log p(x_i) under the mixture model.
    """
    X = np.asarray(X, dtype=float)          # (N,)
    means = np.asarray(means, dtype=float)  # (K,)
    stds = np.asarray(stds, dtype=float)    # (K,)
    mix_probs = np.asarray(mix_probs, dtype=float)  # (K,)
    # safety
    stds = np.maximum(stds, eps)
    mix_probs = mix_probs / mix_probs.sum()
    N = X.shape[0]
    K = means.shape[0]
    # reshape for broadcasting:
    # X: (N, 1), means/stds/mix_probs: (1, K)
    X_expanded = X[:, None]        # (N, 1)
    means_expanded = means[None]   # (1, K)
    stds_expanded = stds[None]     # (1, K)
    mix_expanded = mix_probs[None] # (1, K)
    var = stds_expanded ** 2
    # log p(x_i, z=k) = log pi_k + log N(x_i | mu_k, sigma_k^2)
    log_norm_const = -0.5 * np.log(2.0 * np.pi * var)               # (1, K)
    log_exp_term = -0.5 * (X_expanded - means_expanded) ** 2 / var  # (N, K)
    log_comp = log_norm_const + log_exp_term                        # (N, K)
    log_joint = np.log(mix_expanded + eps) + log_comp               # (N, K)
    # log p(x_i) via log-sum-exp over k
    log_joint_max = np.max(log_joint, axis=1, keepdims=True)        # (N, 1)
    joint_shifted = np.exp(log_joint - log_joint_max)               # (N, K)
    sum_joint = np.sum(joint_shifted, axis=1, keepdims=True)        # (N, 1)
    log_px = (log_joint_max + np.log(sum_joint + eps)).ravel()      # (N,)
    # responsibilities p(z=k | x_i)
    responsibilities = joint_shifted / (sum_joint + eps)            # (N, K)
    # Hard assignments
    labels = np.argmax(responsibilities, axis=1)                    # (N,)
    return labels, responsibilities, log_px
