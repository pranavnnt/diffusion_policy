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
            val_ratio=0.0
            ):
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=[obs_key, action_key])

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
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

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
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

    def _sample_to_data(self, sample):
        # Rename to the standard keys the policy expects:
        obs = sample[self.obs_key]        # shape [T, D_o]
        act = sample[self.action_key]     # shape [T, D_a]
        
        assert obs.ndim == 2, f"Expected obs to be 2D, got {obs.ndim}D"
        assert obs.shape[1] == 37, f"Expected obs to have 37 dimensions, got {obs.shape[1]}"
        
        obs_trimmed = obs_trimmed = np.concatenate([obs[:, :18], obs[:, 24:]], axis=1)
         # Remove forearm and backarm position from state
        assert obs_trimmed.shape[1] == 31, f"Expected trimmed obs to have 31 dimensions, got {obs_trimmed.shape[1]}"
        
        return {
            'obs':    obs_trimmed,
            'action': act,
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)

        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
