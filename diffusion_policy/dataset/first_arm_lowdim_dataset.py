# diffusion_policy/env/first_arm_lowdim_task.py
from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset

class FirstArmLowdimDataset(BaseLowdimDataset):
    def __init__(self, 
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            obs_key='state',
            obs_eef_target=False,
            action_key='action',
            use_manual_normalizer=False,
            seed=42,
            val_ratio=0.0,
            upsampled=True,
            upsample_multiplier=5, 
            ):
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=[obs_key, action_key])

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        
        seq_len = horizon * upsample_multiplier if upsampled else horizon

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=seq_len,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        
        self.obs_key = obs_key
        self.action_key = action_key
        self.use_manual_normalizer = use_manual_normalizer
        self.train_mask = train_mask
        self.obs_eef_target = obs_eef_target
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

        self.upsampled = upsampled
        self.upsample_multiplier = upsample_multiplier

    def get_validation_dataset(self):
        val_set = copy.deepcopy(self)
        seq_len = self.horizon * self.upsample_multiplier if self.upsampled else self.horizon

        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=seq_len,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='gaussian', **kwargs):
        # Build a multi-field normalizer over exactly the keys we will normalize
        data = self._sample_to_data(self.replay_buffer)
        # data must be a dict like {'obs': np.ndarray, 'action': np.ndarray}
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer[self.action_key])

    def __len__(self) -> int:
        return len(self.sampler)
    
    def _filter_obs(self, obs):
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

        force_mag = force[:, 0:1]
        force_vec = force_mag * force[:, 1:4]     

        obs_filtered = np.concatenate(
            [rel_pos_x, rel_pos_z, vel_x, vel_z, cloth_rel_pos_z, cloth_rel_pos_x, force_vec],
            axis=1
        )   # [T, 1+1+1+1+2+5+3 = 14]
        return obs_filtered

    def _sample_to_data(self, sample):
        # Rename to the standard keys the policy expects:

        obs = sample[self.obs_key]        # shape [T, D_o]
        act = sample[self.action_key]     # shape [T, D_a]
        
        assert obs.ndim == 2, f"Expected obs to be 2D, got {obs.ndim}D"
        assert obs.shape[1] == 37, f"Expected obs to have 37 dimensions, got {obs.shape[1]}"
        
        obs_trimmed = np.array(self._filter_obs(obs))
        # Remove forearm and backarm position from state
        assert obs_trimmed.shape[1] == 14, f"Expected trimmed obs to have 14 dimensions, got {obs_trimmed.shape[1]}"
        
        act_trimmed = act[:, [0, 2]]

        return {
            'obs':    obs_trimmed,
            'action': act_trimmed,
        }

    def add_noise(self, obs):
    
        obs_vec = obs["obs"]        # shape (T, 14)
        T = obs_vec.shape[0]

        # --- Noise scales ---
        rel_pos_std              = np.array([1, 1], dtype=np.float32)
        vel_std                  = np.array([0.5, 0.5], dtype=np.float32)
        cloth_rel_pos_z_std      = np.array([2, 2], dtype=np.float32)
        cloth_rel_pos_x_std      = np.array([1, 1, 1, 1, 1], dtype=np.float32)
        force_vec_std            = np.array([2, 2, 2], dtype=np.float32)

        # iid per timestep
        rel_pos_noise         = np.random.normal(0, rel_pos_std,        size=(T, 2))
        vel_noise             = np.random.normal(0, vel_std,            size=(T, 2))
        cloth_rel_pos_z_noise = np.random.normal(0, cloth_rel_pos_z_std,size=(T, 2))
        force_vec_noise       = np.random.normal(0, force_vec_std,      size=(T, 3))

        # one noise sample reused for all timesteps IN THIS SAMPLE (T)
        cloth_rel_pos_x_noise = np.random.normal(0, cloth_rel_pos_x_std)   # (5,)
        cloth_rel_pos_x_noise = np.tile(cloth_rel_pos_x_noise, (T, 1))     # (T, 5)

        noise = np.concatenate([
            rel_pos_noise,
            vel_noise,
            cloth_rel_pos_z_noise,
            cloth_rel_pos_x_noise,
            force_vec_noise
        ], axis=1)

        obs["obs"] = obs_vec + noise
        return obs

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.upsampled:
            raw_sample = self.sampler.sample_sequence(idx)
            # Subsample every `upsample_multiplier` frame to restore original timing
            for k in [self.obs_key, self.action_key]:
                raw_sample[k] = raw_sample[k][::self.upsample_multiplier]
        else:
            raw_sample = self.sampler.sample_sequence(idx)

        data = self._sample_to_data(raw_sample)
        data = self.add_noise(data)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
