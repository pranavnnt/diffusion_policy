import os
import zarr
import torch
import numpy as np
import copy

from typing import Dict, Optional
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset


class DressingRealDataset(BaseLowdimDataset):
    """Simplified dataset for dressing task with dual observations.
    
    Supports both state observations and distilled visual features.
    Based on PushT dataset pattern for clarity.
    """
    
    def __init__(
        self,
        zarr_path: str,
        horizon: int = 1,
        pad_before: int = 0,
        pad_after: int = 0,
        obs_key: str = 'state',
        action_key: str = 'action',
        distilled_features_key: str = 'distilled_features',
        seed: int = 42,
        val_ratio: float = 0.0,
        max_train_episodes: Optional[int] = None
    ):
        super().__init__()
        
        # Build keys to load
        keys = [obs_key, action_key]
        keys.append(distilled_features_key)
        
        # Load replay buffer
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=keys)
        
        # Create train/val split
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed)
        
        # Create sampler
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask
        )
        
        # Store configuration
        self.obs_key = obs_key
        self.action_key = action_key
        self.distilled_features_key = distilled_features_key
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        """Create validation dataset using val_mask."""
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

    def get_normalizer(self, mode: str = 'limits', **kwargs) -> LinearNormalizer:
        """Compute normalizer statistics from entire replay buffer.
        
        Args:
            mode: Normalization mode ('limits' supported)
            
        Returns:
            LinearNormalizer with separate normalizers for 'obs', 'action',
            and optionally 'distilled_features'
        """
        data = self._sample_to_data(self.replay_buffer)
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        """Get all actions from replay buffer."""
        return torch.from_numpy(self.replay_buffer[self.action_key])

    def __len__(self) -> int:
        """Total number of samples."""
        return len(self.sampler)

    def _sample_to_data(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Convert raw sample to processed data dictionary.
        
        Args:
            sample: Raw sample from replay buffer
            
        Returns:
            Dictionary with 'obs', 'action', and optionally 'distilled_features'
        """
        
        state = sample[self.obs_key]
        action = sample[self.action_key]  
        distilled_features = sample[self.distilled_features_key]

        # Build output dictionary
        data = {
            'obs': distilled_features,
            'action': action,
            'state': state
        }
        
        # Add distilled features if enabled
        distilled_features = sample[self.distilled_features_key]
        data['distilled_features'] = distilled_features
        
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with torch tensors for 'obs', 'action',
            and optionally 'distilled_features'
        """
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data


if __name__ == '__main__':
    # Test the dataset
    dataset = DressingRealDataset(
        zarr_path='/scratch/pnt8/dressing_demos/sim2real/real/round3+4_data_green_tee_upsampled_x2.zarr',
        horizon=16,
        pad_before=0,
        pad_after=0,
        obs_key='state',
        action_key='action',
        distilled_features_key='distilled_features',
        seed=42,
        val_ratio=0.1,
        max_train_episodes=None
    )
    
    print(f"Dataset length: {len(dataset)}")
    print(f"Number of episodes: {dataset.replay_buffer.n_episodes}")
    
    # Test sample
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"obs shape: {sample['obs'].shape}")
    print(f"action shape: {sample['action'].shape}")
    if 'distilled_features' in sample:
        print(f"distilled_features shape: {sample['distilled_features'].shape}")
    
    # Test normalizer
    normalizer = dataset.get_normalizer()
    print(f"\nNormalizer keys: {normalizer.keys()}")
    
    # Test validation split
    val_dataset = dataset.get_validation_dataset()
    print(f"\nValidation dataset length: {len(val_dataset)}")