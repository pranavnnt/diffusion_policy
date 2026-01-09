import numpy as np
from typing import Dict, List

from diffusion_policy.dressing.keys import get_distilled_feature_dims, get_state_dict

# Noise standard deviations per feature type
DISTILLED_NOISE_STD = {
    'mask': 0.0,
    'ratio': 0.1,
    'default': 0.02  # For relative position features (3D)
}

STATE_NOISE_STD = {
    'pos': 0.002,
    'vel': 0.01,
    'force': 2.0,
    'default': 0.01
}


def generate_distilled_features_noise(keys: List[str], shape: tuple, dataset_name: str) -> np.ndarray:
    """
    Generate noise for distilled features based on keys.
    
    Args:
        keys: List of distilled feature keys
        shape: Shape of distilled_features array [T, 66] or [B, T, 66]
        dataset_name: Name of dataset
        
    Returns:
        Noise array matching shape
    """
    
    dims = get_distilled_feature_dims(keys)
    noise_components = []
    
    for key, dim in zip(keys, dims):
        suffix = key.split("_")[-1]
        
        # Determine noise std based on suffix
        if suffix == "mask":
            std = DISTILLED_NOISE_STD['mask']
        elif suffix == "ratio":
            std = DISTILLED_NOISE_STD['ratio']
        else:
            std = DISTILLED_NOISE_STD['default']
        
        # Generate noise with appropriate shape
        noise_shape = shape[:-1] + (dim,)  # Replace last dim with feature dim
        noise = np.random.normal(0, std, size=noise_shape)
        noise_components.append(noise)

    noise_matrix = np.concatenate(noise_components, axis=-1).astype(np.float32)
    print("Generated distilled features noise with shape:", noise_matrix.shape)
    
    return noise_matrix


def generate_state_noise(keys: List[str], shape: tuple, dataset_name: str) -> np.ndarray:
    """
    Generate noise for state features based on keys.
    
    Args:
        keys: List of state feature keys
        shape: Shape of state array [T, 18] or [B, T, 18]
        dataset_name: Name of dataset
        
    Returns:
        Noise array matching shape
    """
    if not dataset_name.startswith("sim"):
        return np.zeros(shape, dtype=np.float32)
    
    noise_components = []
    
    for key in keys:
        # Determine noise std based on key type
        if 'pos' in key.lower():
            std = STATE_NOISE_STD['pos']
        elif 'vel' in key.lower():
            std = STATE_NOISE_STD['vel']
        elif 'force' in key.lower():
            std = STATE_NOISE_STD['force']
        else:
            std = STATE_NOISE_STD['default']
        
        # Each state key has dimension 3
        noise_shape = shape[:-1] + (3,)
        noise = np.random.normal(0, std, size=noise_shape)
        noise_components.append(noise)
    
    noise_matrix = np.concatenate(noise_components, axis=-1).astype(np.float32)
    print("Generated state noise with shape:", noise_matrix.shape)
    
    return noise_matrix


def add_noise(
    data: Dict[str, np.ndarray],
    dataset_name: str,
    distilled_keys: List[str] = None,
    state_keys: List[str] = None
) -> Dict[str, np.ndarray]:
    """
    Add key-based noise to state and distilled_features for data augmentation.
    
    Args:
        data: Dictionary containing:
            - 'state': [T, 18] or [B, T, 18] state observations
            - 'distilled_features': [T, 66] or [B, T, 66] visual features
            - 'action': [T, D_a] or [B, T, D_a] actions
        dataset_name: Name of dataset (e.g., 'sim', 'real_first_arm')
        distilled_keys: List of distilled feature keys (required for noise generation)
        state_keys: List of state keys (required for noise generation)
        
    Returns:
        Modified data dictionary with noise added
    """
    # Only add noise to sim data
    if not dataset_name.startswith("sim"):
        return data
    
    # Add noise to state
    if 'state' in data and state_keys is not None:
        state = data['state']
        state_noise = generate_state_noise(state_keys, state.shape, dataset_name)
        data['state'] = state + state_noise
    
    # Add noise to distilled features
    if 'distilled_features' in data and distilled_keys is not None:
        distilled = data['distilled_features']
        distilled_noise = generate_distilled_features_noise(distilled_keys, distilled.shape, dataset_name)
        data['distilled_features'] = distilled + distilled_noise
    
    return data

def filter_state(state: np.ndarray, keys: list[str]) -> np.ndarray:
    """
    Given a full state array and a list of keys, return a filtered state array containing only the features corresponding to the keys.
    """
    dims = [3 for _ in keys]  # All state keys have dimension 3
    filtered_state = []
    index = 0
    for dim in dims:
        filtered_state.append(state[index:index+dim])
        index += dim
    return np.concatenate(filtered_state)

