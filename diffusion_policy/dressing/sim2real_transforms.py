import numpy as np
from typing import Dict

# Constants for better maintainability
SCALING_FACTORS = {
    'rel_pos_x': -180,
    # 'rel_pos_y': 180,
    'rel_pos_z': 180,
    'vel_x': -180,
    # 'vel_y': 180,
    'vel_z': 180,
    'cloth_rel_pos_x': -180,  # applied to 5 dimensions
    'cloth_rel_pos_z': 180,   # applied to 2 dimensions
    'cloth_spread': 180,
    'hand_spread': 180,
    'force_vec': 1,           # applied to 3 dimensions
    'visible_hand_ratio': 1,  # unused
}

SIM_NOISE_STD = {
    'rel_pos': np.array([1, 1], dtype=np.float32),
    'vel': np.array([0.5, 0.5], dtype=np.float32),
    'cloth_rel_pos_x': np.array([1, 1, 1, 1, 1], dtype=np.float32),
    'cloth_rel_pos_z': np.array([2, 2], dtype=np.float32),
    'cloth_spread': np.array([2], dtype=np.float32),
    'hand_spread': np.array([1], dtype=np.float32),
    'force_vec': np.array([2, 2, 2], dtype=np.float32),
    'visible_hand_ratio': np.array([0.05], dtype=np.float32),
}

REAL_NOISE_STD = {
    'rel_pos': np.array([0.005, 0.002], dtype=np.float32),
    'vel': np.array([0.001, 0.001], dtype=np.float32),
    'cloth_rel_pos_x': np.array([0.002, 0.002, 0.002, 0.002, 0.0002], dtype=np.float32),
    'cloth_rel_pos_z': np.array([0.005, 0.001], dtype=np.float32),
    'cloth_spread': np.array([0.005], dtype=np.float32),
    'hand_spread': np.array([0.002], dtype=np.float32),
    'force_vec': np.array([0.01, 0.01, 0.01], dtype=np.float32),
    'visible_hand_ratio': np.array([0.02], dtype=np.float32),
}

def _extract_observation_components(obs: np.ndarray) -> Dict[str, np.ndarray]:
    """Extract individual components from raw observation array.
    
    Args:
        obs: Raw observation array of shape (T, 67+)
        
    Returns:
        Dictionary containing extracted components
    """
    return {
        'pos': obs[:, :3],
        'vel': obs[:, 3:6],
        'in_arm': obs[:, 6],
        'force': obs[:, 7:11],
        'bigger_hole_area': obs[:, 11:12],
        'arm_pos': obs[:, 12:24],
        'hand_pos': obs[:, 24:31],
        'cloth_features': obs[:, 31:37],
        'visible_hand_ratio': obs[:, 37:38]
    }


def _compute_relative_positions(pos: np.ndarray, vel: np.ndarray, 
                                arm_pos: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute relative positions and velocities between fingertip and end-effector."""
    return {
        'rel_pos_x': np.expand_dims(pos[:, 0] - arm_pos[:, 0], axis=1),
        # 'rel_pos_y': np.expand_dims(pos[:, 1] - arm_pos[:, 1], axis=1),
        'rel_pos_z': np.expand_dims(pos[:, 2] - arm_pos[:, 2], axis=1),
        'vel_x': np.expand_dims(vel[:, 0], axis=1),
        # 'vel_y': np.expand_dims(vel[:, 1], axis=1),
        'vel_z': np.expand_dims(vel[:, 2], axis=1)
    }


def _compute_cloth_hand_features(cloth_features: np.ndarray, 
                                 hand_pos: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute relative positions between cloth and hand features.
    
    Note: In simulation, dressing occurs in negative x direction (max x = farthest from dressed).
          In real world, dressing occurs in positive x direction.
    """
    # Cloth relative position in Z (vertical)
    cloth_rel_pos_z = np.stack([
        cloth_features[:, 4] - hand_pos[:, 1],  # min_z - hand_z_min
        cloth_features[:, 5] - hand_pos[:, 0]   # max_z - hand_z_max
    ], axis=1)
    
    # Cloth relative position in X (horizontal) for each finger
    cloth_max_x = cloth_features[:, 1]
    cloth_rel_pos_x = np.stack([
        cloth_max_x - hand_pos[:, 2],  # thumb
        cloth_max_x - hand_pos[:, 3],  # index
        cloth_max_x - hand_pos[:, 4],  # middle
        cloth_max_x - hand_pos[:, 5],  # ring
        cloth_max_x - hand_pos[:, 6],  # pinky
    ], axis=1)
    
    # Spread metrics
    cloth_spread = (cloth_features[:, 5] - cloth_features[:, 4]).reshape(-1, 1)
    hand_spread = (hand_pos[:, 0] - hand_pos[:, 1]).reshape(-1, 1)
    
    return {
        'cloth_rel_pos_x': cloth_rel_pos_x,
        'cloth_rel_pos_z': cloth_rel_pos_z,
        'cloth_spread': cloth_spread,
        'hand_spread': hand_spread
    }


def _compute_force_vector(force: np.ndarray) -> np.ndarray:
    """Compute force vector from magnitude and direction."""
    force_mag = force[:, 0:1]
    force_vec = force_mag * force[:, 1:4]
    return force_vec


def filter_sim_obs(obs: np.ndarray) -> np.ndarray:
    """Filter and extract relevant features from simulation observations.
    
    Args:
        obs: Raw observation array of shape (T, 67+)
        
    Returns:
        Filtered observation array of shape (T, 18)
    """
    components = _extract_observation_components(obs)
    
    rel_features = _compute_relative_positions(
        components['pos'], 
        components['vel'], 
        components['arm_pos']
    )
    
    cloth_hand_features = _compute_cloth_hand_features(
        components['cloth_features'],
        components['hand_pos']
    )
    
    force_vec = _compute_force_vector(components['force'])
    
    # Concatenate all filtered features
    obs_filtered = np.concatenate([
        rel_features['rel_pos_x'],
        rel_features['rel_pos_z'],
        rel_features['vel_x'],
        rel_features['vel_z'],
        cloth_hand_features['cloth_rel_pos_x'],
        cloth_hand_features['cloth_rel_pos_z'],
        cloth_hand_features['cloth_spread'],
        cloth_hand_features['hand_spread'],
        force_vec,
        components['visible_hand_ratio'],  
    ], axis=1)

    assert obs_filtered.shape[1] == 17
    
    return obs_filtered

def filter_real_obs(obs: np.ndarray) -> np.ndarray:

    # assert obs.shape[1] == 38, f"Expected real obs to have 38 dimensions, got {obs.shape[1]}"

    obs_front_only = obs[:, :19]        # using front data only

    # State vector layout from thread_arm_env.py:
    # 0-2: pos (x,y,z), 3-5: vel (x,y,z), 6-10: cloth_rel_pos_x (5),
    # 11-12: cloth_rel_pos_z (2), 13: cloth_spread, 14: hand_spread,
    # 15-17: force (3), 18: visible_hand_ratio
    pos = obs_front_only[:, :3]
    vel = obs_front_only[:, 3:6]
    cloth_rel_pos_x = obs_front_only[:, 6:11]      # unused but extracted for clarity
    cloth_rel_pos_z = obs_front_only[:, 11:13]
    cloth_spread = obs_front_only[:, 13:14]
    hand_spread = obs_front_only[:, 14:15]         # unused
    force = obs_front_only[:, 15:18]
    visible_hand_ratio = obs_front_only[:, 18:19]

    pos_xz = pos[:, [0, 2]]
    vel_xz = vel[:, [0, 2]]

    obs_filtered =  np.concatenate([
        pos_xz,
        vel_xz,
        cloth_rel_pos_x,
        cloth_rel_pos_z,
        cloth_spread,
        hand_spread,
        force,
        visible_hand_ratio
    ], axis=1)
    assert obs_filtered.shape[1] == 17, f"Expected filtered real obs to have 17 dimensions, got {obs_filtered.shape[1]}"

    return obs_filtered

def _build_scaling_vector() -> np.ndarray:
    """Build the scaling vector for observations."""
    vec = np.array([
        SCALING_FACTORS['rel_pos_x'],
        SCALING_FACTORS['rel_pos_z'],
        SCALING_FACTORS['vel_x'],
        SCALING_FACTORS['vel_z'],
        *[SCALING_FACTORS['cloth_rel_pos_x']] * 5,
        *[SCALING_FACTORS['cloth_rel_pos_z']] * 2,
        SCALING_FACTORS['cloth_spread'],
        SCALING_FACTORS['hand_spread'],
        *[SCALING_FACTORS['force_vec']] * 3,
        SCALING_FACTORS['visible_hand_ratio'],
    ])

    assert vec.shape[0] == 17, f"Expected scaling vector to have 17 dimensions, got {vec.shape[0]}"
    return vec


def scale_sim_obs(obs: np.ndarray) -> np.ndarray:
    """Scale filtered simulation observations."""
    scaling_vector = _build_scaling_vector()

    return obs / scaling_vector


def scale_noise(noise: np.ndarray) -> np.ndarray:
    """Scale noise using the same scaling as observations."""
    return scale_sim_obs(noise)


def scale_sim_action(action_trimmed: np.ndarray) -> np.ndarray:
    """Scale simulation actions."""
    
    scaling_vector = np.array([
        SCALING_FACTORS['vel_x'],  # action[0]: x direction (flipped)
        # SCALING_FACTORS['vel_y'],      # action[1]: y direction (added)
        SCALING_FACTORS['vel_z']   # action[1]: z direction (same)
    ])
    return action_trimmed / scaling_vector


def _generate_noise(timesteps: int, noise_std) -> np.ndarray:
    """Generate noise for observations.
    
    Args:
        timesteps: Number of timesteps (T)
        
    Returns:
        Noise array of shape (T, 18)
    """
    # Independent noise per timestep
    vel_noise = np.random.normal(0, noise_std['vel'], size=(timesteps, 2))
    cloth_rel_pos_z_noise = np.random.normal(0, noise_std['cloth_rel_pos_z'], size=(timesteps, 2))
    cloth_spread_noise = np.random.normal(0, noise_std['cloth_spread'], size=(timesteps, 1))
    force_vec_noise = np.random.normal(0, noise_std['force_vec'], size=(timesteps, 3))
    visible_hand_ratio_noise = np.random.normal(0, noise_std['visible_hand_ratio'], size=(timesteps, 1))

    # Shared noise across all timesteps (one sample per episode)

    rel_pos_noise = np.tile(
        np.random.normal(0, noise_std['rel_pos'], size=(1, 2)),
        (timesteps, 1)
    )

    cloth_rel_pos_x_noise = np.tile(
        np.random.normal(0, noise_std['cloth_rel_pos_x'], size=(1, 5)),
        (timesteps, 1)
    )
    hand_spread_noise = np.tile(
        np.random.normal(0, noise_std['hand_spread']),
        (timesteps, 1)
    )

    noise = np.concatenate([
        rel_pos_noise,
        vel_noise,
        cloth_rel_pos_x_noise,
        cloth_rel_pos_z_noise,
        cloth_spread_noise,
        hand_spread_noise,
        force_vec_noise,
        visible_hand_ratio_noise
    ], axis=1)

    assert noise.shape == (timesteps, 17), f"Expected noise shape to be {(timesteps, 16)}, got {noise.shape}"
    
    return noise


def add_noise(obs: Dict[str, np.ndarray], dataset_name: str) -> Dict[str, np.ndarray]:
    """Add scaled noise to observations (works for both scaled sim and real observations).
    
    Args:
        obs: Dictionary containing 'obs' key with array of shape (T, 38)
        
    Returns:
        Modified observation dictionary with noise added
    """

    obs_vec = obs["obs"]
    timesteps = obs_vec.shape[0]
    
    if dataset_name.startswith("sim"):
        noise = _generate_noise(timesteps, SIM_NOISE_STD)
        scaled_noise = scale_noise(noise)
    else:

        if obs_vec.shape[1] == 38:
            noise1 = _generate_noise(timesteps, SIM_NOISE_STD)
            scaled_noise1 = scale_noise(noise1)
            noise2 = _generate_noise(timesteps, REAL_NOISE_STD)
            scaled_noise2 = noise2
            scaled_noise = np.concatenate([scaled_noise1, scaled_noise2], axis=1)
        else:
            noise = _generate_noise(timesteps, REAL_NOISE_STD)
            scaled_noise = noise
    
    obs["obs"] = obs_vec + scaled_noise
    return obs