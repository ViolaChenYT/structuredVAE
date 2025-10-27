#!/usr/bin/env python3
"""
Test script for coordinate visualization functionality.

This script tests the new coordinate visualization functions that work directly
with AnnData objects instead of CSV files.
"""

import sys
import os
sys.path.append('/n/fs/ragr-data/users/viola/structuredVAE/scripts')

from analyze_gmm_comparison import (
    visualize_coordinates_from_adata,
    visualize_multiple_paths_coordinates,
    create_coordinate_visualizations
)

def test_single_path_visualization():
    """Test visualization for a single path."""
    base_dir = "/n/fs/ragr-data/users/viola/structuredVAE/data"
    
    # Test with a specific path (you can change this to any path you have)
    test_path = "MS_MSaapapaa"  # Example path name
    
    print(f"Testing single path visualization for: {test_path}")
    
    # Test PCA visualization
    fig_pca = visualize_coordinates_from_adata(
        test_path, base_dir, method='pca', figsize=(10, 6)
    )
    
    if fig_pca is not None:
        print("✓ PCA visualization created successfully")
        # Save the figure
        output_dir = "/n/fs/ragr-data/users/viola/structuredVAE/results/visualizations"
        os.makedirs(output_dir, exist_ok=True)
        fig_pca.savefig(f"{output_dir}/test_pca_{test_path}.png", dpi=300, bbox_inches='tight')
        print(f"PCA visualization saved to: {output_dir}/test_pca_{test_path}.png")
    else:
        print("✗ PCA visualization failed")
    
    # Test UMAP visualization
    fig_umap = visualize_coordinates_from_adata(
        test_path, base_dir, method='umap', figsize=(10, 6)
    )
    
    if fig_umap is not None:
        print("✓ UMAP visualization created successfully")
        fig_umap.savefig(f"{output_dir}/test_umap_{test_path}.png", dpi=300, bbox_inches='tight')
        print(f"UMAP visualization saved to: {output_dir}/test_umap_{test_path}.png")
    else:
        print("✗ UMAP visualization failed")

def test_multiple_paths_visualization():
    """Test visualization for multiple paths."""
    base_dir = "/n/fs/ragr-data/users/viola/structuredVAE/data"
    
    # Get a few test paths
    import glob
    loom_files = glob.glob(os.path.join(base_dir, "*.loom"))
    path_names = [os.path.basename(f).replace('.loom', '') for f in loom_files[:5]]  # First 5 paths
    
    print(f"Testing multiple paths visualization for: {path_names}")
    
    # Test PCA visualization
    fig_pca = visualize_multiple_paths_coordinates(
        base_dir, path_names, method='pca', n_cols=3, figsize=(15, 10)
    )
    
    if fig_pca is not None:
        print("✓ Multiple paths PCA visualization created successfully")
        output_dir = "/n/fs/ragr-data/users/viola/structuredVAE/results/visualizations"
        os.makedirs(output_dir, exist_ok=True)
        fig_pca.savefig(f"{output_dir}/test_multiple_pca.png", dpi=300, bbox_inches='tight')
        print(f"Multiple paths PCA visualization saved to: {output_dir}/test_multiple_pca.png")
    else:
        print("✗ Multiple paths PCA visualization failed")

def test_coordinate_visualization_function():
    """Test the main coordinate visualization function."""
    base_dir = "/n/fs/ragr-data/users/viola/structuredVAE/data"
    output_dir = "/n/fs/ragr-data/users/viola/structuredVAE/results/visualizations"
    
    print("Testing main coordinate visualization function...")
    
    # Test PCA visualization
    try:
        create_coordinate_visualizations(
            base_dir, n_paths=10, method='pca', output_dir=output_dir
        )
        print("✓ PCA coordinate visualization function completed successfully")
    except Exception as e:
        print(f"✗ PCA coordinate visualization function failed: {e}")
    
    # Test UMAP visualization
    try:
        create_coordinate_visualizations(
            base_dir, n_paths=10, method='umap', output_dir=output_dir
        )
        print("✓ UMAP coordinate visualization function completed successfully")
    except Exception as e:
        print(f"✗ UMAP coordinate visualization function failed: {e}")

def main():
    """Run all tests."""
    print("="*60)
    print("TESTING COORDINATE VISUALIZATION FUNCTIONALITY")
    print("="*60)
    
    print("\n1. Testing single path visualization...")
    test_single_path_visualization()
    
    print("\n2. Testing multiple paths visualization...")
    test_multiple_paths_visualization()
    
    print("\n3. Testing main coordinate visualization function...")
    test_coordinate_visualization_function()
    
    print("\n" + "="*60)
    print("TESTING COMPLETE")
    print("="*60)

if __name__ == '__main__':
    main()



