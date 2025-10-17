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
            depth_key='depth',
            use_manual_normalizer=False,
            modalities=['rgb', 'pos', 'vel', 'force', 'depth'],
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
        self.depth_key = depth_key
        self.use_manual_normalizer = use_manual_normalizer
        self.train_mask = train_mask
        self.obs_eef_target = obs_eef_target
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.modalities = modalities
        
        print("Modalities used for training: ", self.modalities)
        
        assert "rgb" in self.modalities, "rgb modality is required"
        
        print("FirstArmHybridDataset initialized.")
        print("Code set up to take obs dim 37, remove arm states to return privileged state with obs dim 31")
        print("Actual state (without privileged information) varies, depending on modalities chosen.")

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
        
        state = self.replay_buffer[self.obs_key]
        state = self._filter_obs(state)
        
        data = {
            'action': self.replay_buffer[self.action_key],
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
    
    def _filter_obs(self, obs):
        
        pos = obs[:, :3]
        vel = obs[:, 3:6]
        force = obs[:, 7:11]
        
        obs_filtered = []
        if 'pos' in self.modalities:
            obs_filtered.append(pos)
        if 'vel' in self.modalities:
            obs_filtered.append(vel)
        if 'force' in self.modalities:
            obs_filtered.append(force)
        
        assert len(obs_filtered) > 0, "At least one of pos, vel, force must be in modalities (for now!)"
        obs_filtered = np.concatenate(obs_filtered, axis=1)
        
        return obs_filtered

    def _sample_to_data(self, sample):
        obs   = sample[self.obs_key][...].astype(np.float32)   # (T, 37)
        act   = sample[self.action_key][...].astype(np.float32)  # (T, 3)
        img   = sample[self.img_key][...].astype(np.float32) / 255.0  # (T, 3, 128, 128)
        depth = sample[self.depth_key][...].astype(np.float32)     # (T, 1, 128, 128)    
        
        assert depth.min() >= 0.0 and depth.max() <= 1.0, "Depth should be already be normalized to [0, 1]"
        
        # trim full privileged state info based on modalities
        obs_filtered = self._filter_obs(obs)  # dimension of axis 1 varies based on modalities
        
        if "depth" in self.modalities:
            output_img = np.concatenate([img, depth], axis=1)
        else:
            output_img = img
        
        return {
            'obs': {
                self.img_key: output_img,
                self.obs_key: obs_filtered,
            },
            self.action_key: act
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)

        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
    
    def get_shape_meta(self):
        obs_shape = self._filter_obs(self.replay_buffer[self.obs_key][:1]).shape[-1]
        img = self.replay_buffer[self.img_key][0]
        depth = self.replay_buffer[self.depth_key][0]
        img_shape = img.shape if "depth" not in self.modalities else (img.shape[0] + depth.shape[0], *img.shape[1:])
        return {
            "obs": {
                "state": {"shape": (obs_shape,)},
                "image": {"shape": img_shape},
            },
            "action": {"shape": self.replay_buffer[self.action_key][0].shape},
        }

    def preprocess_obs(self, sample):
        """Process a single Unity observation dict into tensors."""
        img = sample[self.img_key]          # RGB
        depth = sample.get(self.depth_key, None)
        obs = sample[self.obs_key]
    
        # --- Ensure CHW format ---
        if img.ndim == 3 and img.shape[0] != 3:
            img = img.transpose(2, 0, 1)
        if depth is not None and depth.ndim == 2:
            depth = depth[None, :, :]
    
        # --- Combine RGB + Depth ---
        if "depth" in self.modalities:
            output_img = np.concatenate([img, depth], axis=0)
        else:
            output_img = img
    
        # --- Resize & normalize ---
        output_img = cv2.resize(output_img.transpose(1, 2, 0), (128, 128), interpolation=cv2.INTER_AREA)
        output_img = output_img.transpose(2, 0, 1).astype(np.float32) / 255.0
    
        # --- Filter obs (state) ---
        obs_filtered = self._filter_obs(np.array(obs)[None, :])[0]
    
        return {
            "state": torch.from_numpy(obs_filtered).float(),
            "image": torch.from_numpy(output_img).float(),
        }