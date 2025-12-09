"""
scvi_wae module for Wasserstein Autoencoder with SCVI integration.
"""

from scvi_wae.clustering_utils import gmm_cluster_1d
from scvi_wae.losses import (
    mixture_uniform_reg,
    pairwise_distance_loss,
    wasserstein_distance_1d_learnable,
    wasserstein_distance_1d_mixture_sample,
)
from scvi_wae.path_utils import label_order_index
from scvi_wae.trainer import train_and_eval

__all__ = [
    "gmm_cluster_1d",
    "mixture_uniform_reg",
    "pairwise_distance_loss",
    "wasserstein_distance_1d_learnable",
    "wasserstein_distance_1d_mixture_sample",
    "label_order_index",
    "train_and_eval",
]
