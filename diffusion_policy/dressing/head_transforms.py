import numpy as np
from typing import Dict

NOISE_STD = {
    'rel_pos': np.array([0.01, 0.01, 0.01], dtype=np.float32),
    'vel': np.array([0.001, 0.001, 0.001], dtype=np.float32),
    'force': np.array([1, 1, 1], dtype=np.float32),
    'coverage': np.array([0.02], dtype=np.float32),
    'hull_centroid': np.array([0.01, 0.01, 0.01], dtype=np.float32),
    'hull_area': np.array([0.02], dtype=np.float32),
}

def _extract_observation_components(obs: np.ndarray) -> Dict[str, np.ndarray]:
    """Extract individual components from raw observation array.
    
    Args:
        obs: Raw observation array of shape (T, 67+)
        
    Returns:
        Dictionary containing extracted components
    """

    return {
        'arm1_pos': obs[:, :3],
        'arm1_vel': obs[:, 3:6],
        'arm1_force': obs[:, 6:9],
        'arm2_pos': obs[:, 9:12],
        'arm2_vel': obs[:, 12:15],
        'arm2_force': obs[:, 15:18],
        'head_min_x': obs[:, 18:19],
        'head_max_x': obs[:, 19:20],
        'head_min_y': obs[:, 20:21],
        'head_max_y': obs[:, 21:22],
        'head_min_z': obs[:, 22:23],
        'head_max_z': obs[:, 23:24],
        'face_coverage': obs[:, 24:25],
        'hull_centroid': obs[:, 25:28],
        'hull_area': obs[:, 28:29],
    }

def filter_head_obs(obs: np.ndarray) -> np.ndarray:

    assert obs.shape[1] == 29, f"Expected real obs to have 29 dimensions, got {obs.shape[1]}"

    individual_components = _extract_observation_components(obs)

    arm1_rel_pos_x = individual_components['arm1_pos'][:, 0:1] - individual_components['head_min_x']
    arm1_rel_pos_y = individual_components['arm1_pos'][:, 1:2] - individual_components['head_max_y']
    arm1_rel_pos_z = individual_components['arm1_pos'][:, 2:3] - individual_components['head_max_z']

    arm2_rel_pos_x = individual_components['arm2_pos'][:, 0:1] - individual_components['head_max_x']
    arm2_rel_pos_y = individual_components['arm2_pos'][:, 1:2] - individual_components['head_max_y']
    arm2_rel_pos_z = individual_components['arm2_pos'][:, 2:3] - individual_components['head_max_z']

    coverage = individual_components['face_coverage']
    hull_centroid = individual_components['hull_centroid']
    hull_area = individual_components['hull_area']

    obs_filtered =  np.concatenate([
        arm1_rel_pos_x,
        arm1_rel_pos_y,
        arm1_rel_pos_z,
        individual_components['arm1_vel'],
        individual_components['arm1_force'],
        arm2_rel_pos_x,
        arm2_rel_pos_y,
        arm2_rel_pos_z,
        individual_components['arm2_vel'],
        individual_components['arm2_force'],
        coverage], axis=1)
        # hull_centroid,
        # hull_area]

    assert obs_filtered.shape[1] == 19, f"Expected filtered real obs to have 19 dimensions, got {obs_filtered.shape[1]}"

    return obs_filtered

def _generate_noise(timesteps: int, noise_std) -> np.ndarray:
    """Generate noise for observations """
    # Independent noise per timestep

    arm1_rel_pos_noise = np.random.normal(0, noise_std['rel_pos'], size=(timesteps, 3))
    arm1_vel_noise = np.random.normal(0, noise_std['vel'], size=(timesteps, 3))
    arm1_force_noise = np.random.normal(0, noise_std['force'], size=(timesteps, 3))
    arm2_rel_pos_noise = np.random.normal(0, noise_std['rel_pos'], size=(timesteps, 3))
    arm2_vel_noise = np.random.normal(0, noise_std['vel'], size=(timesteps, 3))
    arm2_force_noise = np.random.normal(0, noise_std['force'], size=(timesteps, 3))
    coverage_noise = np.random.normal(0, noise_std['coverage'], size=(timesteps, 1))
    hull_centroid_noise = np.random.normal(0, noise_std['hull_centroid'], size=(timesteps, 3))
    hull_area_noise = np.random.normal(0, noise_std['hull_area'], size=(timesteps, 1))

    # Shared noise across all timesteps (one sample per episode)

    noise = np.concatenate([
        arm1_rel_pos_noise,
        arm1_vel_noise,
        arm1_force_noise,
        arm2_rel_pos_noise,
        arm2_vel_noise,
        arm2_force_noise,
        coverage_noise,
        # hull_centroid_noise,
        # hull_area_noise
    ], axis=1)

    assert noise.shape == (timesteps, 19), f"Expected noise shape to be {(timesteps, 19)}, got {noise.shape}"
    
    return noise

def add_noise(obs: Dict[str, np.ndarray], dataset_name: str) -> Dict[str, np.ndarray]:
    """Add scaled noise to observations (works for both scaled sim and real observations).
    
    Args:
        obs: Dictionary containing 'obs' key with array of shape (T, D)

        
    Returns:
        Modified observation dictionary with noise added
    """

    obs_vec = obs["obs"]
    timesteps = obs_vec.shape[0]

    noise = _generate_noise(timesteps, NOISE_STD)
    
    obs["obs"] = obs_vec + noise
    return obs