"""
Compute force bin edges from a zarr dataset.

Edit the parameters below and run:
    python compute_force_bins.py
"""

import numpy as np
import zarr
from pathlib import Path
import sys
import os

# Add the parent directory to the path to import sim_transforms
from sim2real_transforms import filter_sim_obs, scale_sim_obs


# ============================================================================
# PARAMETERS - EDIT THESE
# ============================================================================
INPUT_ZARR = '/home/pnt8/workspace/dressing_sim_ws/diffusion_policy/diffusion_policy/data/sim/halton_base_dagger_rd2_x1_aligned_upsampled_x4.zarr'
OUTPUT_BINS = '/home/pnt8/workspace/dressing_sim_ws/diffusion_policy/diffusion_policy/data/sim/sim_force_bins_n10.npz'
NUM_BINS = 10
OBS_KEY = 'data/state'
# ============================================================================


def main():
    print(f"Loading zarr dataset from: {INPUT_ZARR}")
    input_zarr = zarr.open(INPUT_ZARR, mode='r')
    
    # Check available keys
    available_keys = list(input_zarr.keys())
    print(f"Available keys in zarr: {available_keys}")
    
    if OBS_KEY not in input_zarr:
        raise KeyError(f"Key '{OBS_KEY}' not found in zarr file. Available keys: {available_keys}")
    
    obs_data = input_zarr[OBS_KEY][:]
    print(f"Loaded observations with shape: {obs_data.shape}")
    
    # Filter and scale
    print("\nFiltering and scaling observations...")
    obs_filtered = filter_sim_obs(obs_data)
    obs_scaled = scale_sim_obs(obs_filtered)
    
    # Extract force from last 3 dimensions
    force_vec = obs_scaled[:, -3:]
    print(f"Extracted force vectors with shape: {force_vec.shape}")
    
    # Compute statistics
    force_min = np.min(force_vec, axis=0)
    force_max = np.max(force_vec, axis=0)
    force_mean = np.mean(force_vec, axis=0)
    force_std = np.std(force_vec, axis=0)
    
    print("\nForce statistics (after filtering and scaling):")
    print(f"  X-axis: min={force_min[0]:.4f}, max={force_max[0]:.4f}, mean={force_mean[0]:.4f}, std={force_std[0]:.4f}")
    print(f"  Y-axis: min={force_min[1]:.4f}, max={force_max[1]:.4f}, mean={force_mean[1]:.4f}, std={force_std[1]:.4f}")
    print(f"  Z-axis: min={force_min[2]:.4f}, max={force_max[2]:.4f}, mean={force_mean[2]:.4f}, std={force_std[2]:.4f}")
    
    # Create bin edges
    bin_edges = []
    for dim in range(3):
        edges = np.linspace(force_min[dim], force_max[dim], NUM_BINS - 1)
        bin_edges.append(edges)
    
    bin_edges = np.array(bin_edges)
    
    print(f"\nComputed bin edges for {NUM_BINS} bins:")
    print(f"  X-axis bins: {bin_edges[0]}")
    print(f"  Y-axis bins: {bin_edges[1]}")
    print(f"  Z-axis bins: {bin_edges[2]}")
    
    # Print bin distribution
    print("\nBin distribution:")
    for dim, axis_name in enumerate(['X', 'Y', 'Z']):
        bin_indices = np.searchsorted(bin_edges[dim], force_vec[:, dim], side='right')
        counts = np.bincount(bin_indices, minlength=NUM_BINS)
        percentages = 100 * counts / len(force_vec)
        
        print(f"  {axis_name}-axis:")
        for bin_idx in range(NUM_BINS):
            print(f"    Bin {bin_idx}: {counts[bin_idx]:6d} samples ({percentages[bin_idx]:5.2f}%)")
    
    # Save to file
    output_path = Path(OUTPUT_BINS)
    np.savez(
        output_path,
        bin_edges=bin_edges,
        force_min=force_min,
        force_max=force_max,
        force_mean=force_mean,
        force_std=force_std,
        num_samples=force_vec.shape[0],
        num_bins=NUM_BINS
    )
    
    print(f"\n✓ Saved force bins to: {output_path}")
    print(f"  - bin_edges: shape {bin_edges.shape}")
    print(f"  - Statistics: min, max, mean, std")
    print(f"  - Number of samples: {force_vec.shape[0]}")
    
    print("\nTo load in your code:")
    print(f"  data = np.load('{output_path}')")
    print(f"  bin_edges = data['bin_edges']")


if __name__ == '__main__':
    main()