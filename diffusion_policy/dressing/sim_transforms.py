import numpy as np

def _ensure_2d(x):
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[None, :]
    return x.astype(np.float32)

def filter_sim_obs(obs):
        
    pos = obs[:, :3]
    vel = obs[:, 3:6]
    in_arm = obs[:, 6]
    force = obs[:, 7:11]
    bigger_hole_area = obs[:, 11:12]       # keep 2-D shape
    arm_pos = obs[:, 12:24]
    hand_pos = obs[:, 24:31]
    cloth_features = obs[:, 31:]

    # distance between fingertip and EEF in X direction
    rel_pos_x = np.expand_dims(pos[:, 0] - arm_pos[:, 0], axis=1)
    rel_pos_z = np.expand_dims(pos[:, 2] - arm_pos[:, 2], axis=1)
    vel_x     = np.expand_dims(vel[:, 0], axis=1)
    vel_z     = np.expand_dims(vel[:, 2], axis=1)

    # hand_position_state = [hand_z_max, hand_z_min, thumb_x_max, index_x_max, middle_x_max, ring_x_max, pinky_x_max]
    # cloth position state = cloth_features = [min(pos[0] for pos in relative_cloth_loop), max(pos[0] for pos in relative_cloth_loop),
    #                                          min(pos[1] for pos in relative_cloth_loop), max(pos[1] for pos in relative_cloth_loop),
    #                                          min(pos[2] for pos in relative_cloth_loop), max(pos[2] for pos in relative_cloth_loop)]

    # remember, in sim, dressing is in negative x direction, so max x is the farthest from being dresssed. 
    # in the real world, dressing is in positive x direction. 
        
    cloth_rel_pos_z = np.stack([cloth_features[:, 4] - hand_pos[:, 1],
                                cloth_features[:, 5] - hand_pos[:, 0]], axis=1)

    cloth_rel_pos_x = np.stack([cloth_features[:, 1] - hand_pos[:, 2],
                                cloth_features[:, 1] - hand_pos[:, 3], 
                                cloth_features[:, 1] - hand_pos[:, 4],
                                cloth_features[:, 1] - hand_pos[:, 5],
                                cloth_features[:, 1] - hand_pos[:, 6]], axis=1)

    cloth_spread = (cloth_features[:, 5] - cloth_features[:, 4]).reshape(-1, 1)
    hand_spread = (hand_pos[:, 0] - hand_pos[:, 1]).reshape(-1, 1)

    force_mag = force[:, 0:1]
    force_vec = force_mag * force[:, 1:4]     

    obs_filtered = np.concatenate(
        [rel_pos_x, rel_pos_z, vel_x, vel_z, cloth_rel_pos_x, cloth_rel_pos_z, cloth_spread, hand_spread, force_vec],
        axis=1
    )   # [T, 1+1+1+1+2+5+2+3 = 16]
    return obs_filtered

def scale_sim_obs(obs):

    # obs_filtered = np.concatenate(
    #    [rel_pos_x, rel_pos_z, vel_x, vel_z, cloth_rel_pos_x, cloth_rel_pos_z, cloth_spread, hand_spread, force_vec],
    #    axis=1
    #)

    scaling_vector = np.array([-180,                             # rel_pos_x: directions are flipped for x in real world 
                                180,                             # rel_pos_z
                               -180,                             # vel_x
                                180,                             # vel_z
                               -180, -180, -180, -180, -180,      # cloth_rel_pos_x (5)
                                180,  180,                       # cloth_rel_pos_z (2)   
                                180,                             # cloth_spread
                                180,                             # hand_spread
                                1, 1, 1])                        # force_vec (3)

    obs_scaled = obs / scaling_vector

    return obs_scaled

def scale_noise(noise):

    # because the scaling operation is the same for obs and noise,
    return scale_sim_obs(noise)

def scale_sim_action(action_trimmed):

    scaling_vector = np.array([-180,        # action[0]: directions are flipped for x in real world
                                180])       # action[1]: same for z

    action_scaled = action_trimmed / scaling_vector

    return action_scaled

def add_noise(obs):

    # adds noise to both scaled sim and real obs
    
    obs_vec = obs["obs"]        # shape (T, 16)
    T = obs_vec.shape[0]

    # --- Noise scales ---
    rel_pos_std              = np.array([1, 1], dtype=np.float32)
    vel_std                  = np.array([0.5, 0.5], dtype=np.float32)
    cloth_rel_pos_z_std      = np.array([2, 2], dtype=np.float32)
    cloth_rel_pos_x_std      = np.array([1, 1, 1, 1, 1], dtype=np.float32)
    cloth_spread_std         = np.array([2], dtype=np.float32)
    hand_spread_std          = np.array([1], dtype=np.float32)
    force_vec_std            = np.array([2, 2, 2], dtype=np.float32)

    # iid per timestep
    rel_pos_noise         = np.random.normal(0, rel_pos_std,        size=(T, 2))
    vel_noise             = np.random.normal(0, vel_std,            size=(T, 2))
    cloth_rel_pos_z_noise = np.random.normal(0, cloth_rel_pos_z_std,size=(T, 2))
    force_vec_noise       = np.random.normal(0, force_vec_std,      size=(T, 3))
    cloth_spread_noise    = np.random.normal(0, cloth_spread_std,     size=(T, 1))

    # one noise sample reused for all timesteps IN THIS SAMPLE (T)
    cloth_rel_pos_x_noise = np.random.normal(0, cloth_rel_pos_x_std)   # (5,)
    cloth_rel_pos_x_noise = np.tile(cloth_rel_pos_x_noise, (T, 1))     # (T, 5)
    hand_spread_noise    = np.random.normal(0, hand_spread_std)          # (1,)
    hand_spread_noise    = np.tile(hand_spread_noise, (T, 1))           # (T, 1)

    noise = np.concatenate([
        rel_pos_noise,
        vel_noise,
        cloth_rel_pos_x_noise,
        cloth_rel_pos_z_noise,
        cloth_spread_noise,
        hand_spread_noise,
        force_vec_noise
    ], axis=1)

    scaled_noise = scale_noise(noise)

    obs["obs"] = obs_vec + scaled_noise
    return obs