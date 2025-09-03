# diffusion_policy/env/first_arm_lowdim_task.py
from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import (
    get_range_normalizer_from_stat,
    get_image_range_normalizer,
    get_identity_normalizer_from_stat,
    array_to_stats
)
import cv2

class FirstArmHybridDataset(BaseImageDataset):
    def __init__(self, 
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            obs_key='state',
            obs_eef_target=False,
            action_key='action',
            img_key='image',
            use_manual_normalizer=False,
            seed=42,
            val_ratio=0.0
            ):
        super().__init__()
        self.replay_buffer = ReplayBuffer.create_from_path(zarr_path, mode='r')

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
        self.img_key = img_key
        self.use_manual_normalizer = use_manual_normalizer
        self.train_mask = train_mask
        self.obs_eef_target = obs_eef_target
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        
        print("FirstArmHybridDataset initialized.")
        print("Code set up to take obs dim 37, filter out states to return state with obs dim 33")
        print("(removing position of forearm and backarm from state)")

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
        # Sample data dict
        
        state = self.replay_buffer['state']
        state = np.concatenate([state[:, :18], state[:, 24:]], axis=1).astype(np.float32)
        
        data = {
            'action': self.replay_buffer['action'],
            'state': state
        }

        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer["image"] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer[self.action_key])

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        obs = sample[self.obs_key][...].astype(np.float32)   # (T, 37)
        act = sample[self.action_key][...].astype(np.float32)  # (T, 3)
        img = sample[self.img_key][...].astype(np.float32) / 255.0  # (T, 3, 128, 128)
        
        # trim forearm/backarm dims
        obs_trimmed = np.concatenate([obs[:, :18], obs[:, 24:]], axis=1)
    
        return {
            'obs': {
                self.img_key: img,
                self.obs_key: obs_trimmed,
            },
            self.action_key: act
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)

        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
