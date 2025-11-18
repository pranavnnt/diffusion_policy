import numpy as np

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

def scale_sim_action(action_trimmed):

    scaling_vector = np.array([-180,        # action[0]: directions are flipped for x in real world
                                180])       # action[1]: same for z

    action_scaled = action_trimmed / scaling_vector

    return action_scaled