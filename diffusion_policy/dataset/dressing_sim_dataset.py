from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset

from diffusion_policy.dressing.sim2real_transforms import (
    filter_sim_obs,
    scale_sim_obs,
    scale_sim_action,
    add_noise
)


class DressingSimDataset(BaseLowdimDataset):
    def __init__(self, 
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            obs_key='state',
            obs_eef_target=False,
            action_key='action',
            use_manual_normalizer=False,
            use_domain_encoding=True,
            domain_encoding_dim=2,
            seed=42,
            val_ratio=0.0,
            upsampled=True,
            upsample_multiplier=5,
            force_bins_path="/home/pnt8/workspace/dressing_sim_ws/diffusion_policy/diffusion_policy/data/sim/sim_force_bins_n10.npz",
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
        self.use_domain_encoding = use_domain_encoding
        self.domain_encoding_dim = domain_encoding_dim
        self.train_mask = train_mask
        self.obs_eef_target = obs_eef_target
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

        print("Domain encoding set to: ", self.use_domain_encoding)

        self.upsampled = upsampled
        self.upsample_multiplier = upsample_multiplier
        
        # Force binning setup
        self.force_bins = None
        self.force_bin_edges = None
        if force_bins_path is not None:
            self._load_force_bins(force_bins_path)
    
    def _load_force_bins(self, force_bins_path):
        """Load pre-computed force bin edges from file."""
        print(f"Loading force bins from: {force_bins_path}")
        force_data = np.load(force_bins_path)
        self.force_bin_edges = force_data['bin_edges']
        self.force_bins = int(force_data['num_bins'])
        
        print(f"Loaded force binning configuration:")
        print(f"  Number of bins: {self.force_bins}")
        print(f"  X-axis bins: {self.force_bin_edges[0]}")
        print(f"  Y-axis bins: {self.force_bin_edges[1]}")
        print(f"  Z-axis bins: {self.force_bin_edges[2]}")
    
    def discretize_force(self, force_vec):
        """
        Discretize force vector into bins.
        
        Args:
            force_vec: shape [T, 3] - continuous force values
            
        Returns:
            force_bins_indices: shape [T, 3] - bin indices (0 to force_bins-1)
        """
        assert self.force_bins is not None, "force_bins must be set to discretize force"
        assert self.force_bin_edges is not None, "force_bin_edges not loaded"
        
        T = force_vec.shape[0]
        force_bins_indices = np.zeros((T, 3), dtype=np.int64)
        
        for dim in range(3):
            force_bins_indices[:, dim] = np.searchsorted(
                self.force_bin_edges[dim], force_vec[:, dim], side='right'
            )
        
        return force_bins_indices
    
    def apply_force_binning(self, data):
        """Apply force binning to observation after noise has been added."""
        if self.force_bins is not None:
            force_bins_indices = self.discretize_force(data['obs'][:, -3:])
            data['obs'] = data['obs'].copy()
            data['obs'][:, -3:] = force_bins_indices.astype(np.float32)
        return data

    def get_validation_dataset(self):
        """Create a validation dataset using the validation mask."""
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
        """Build a multi-field normalizer over the data keys."""
        data = self._sample_to_data(self.replay_buffer)
        
        # If force binning is enabled, we need to handle normalization carefully
        if self.force_bins is not None:
            # Get force values before binning for proper normalization stats
            # We'll normalize everything except force, then set force normalization to identity
            normalizer = LinearNormalizer()
            normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
            
            # Override normalization for force dimensions (last 3)
            # Set to identity: offset=0, scale=1 (no normalization)
            normalizer['obs'].params_dict['offset'][-3:] = 0.0
            normalizer['obs'].params_dict['scale'][-3:] = 1.0
            
            print("Force binning enabled: force dimensions will not be normalized")
            print(f"  Bins 0 to {self.force_bins-1} will be used as categorical values")
        else:
            # Normal normalization for all dimensions
            normalizer = LinearNormalizer()
            normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        """Get all actions from the replay buffer."""
        return torch.from_numpy(self.replay_buffer[self.action_key])

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Convert raw sample to processed observation and action data.
        
        Args:
            sample: Dictionary containing raw observations and actions
            
        Returns:
            Dictionary with processed 'obs' and 'action' arrays, and optionally 'domain_encoding'
        """
        obs = sample[self.obs_key]  # shape [T, 37]
        act = sample[self.action_key]  # shape [T, D_a]

        # Validate input dimensions
        assert obs.ndim == 2, f"Expected obs to be 2D, got {obs.ndim}D"
        assert obs.shape[1] == 37, f"Expected obs to have 37 dimensions, got {obs.shape[1]}"

        # Filter observations to extract relevant features
        obs_filtered = filter_sim_obs(obs)
        assert obs_filtered.shape[1] == 16, (
            f"Expected filtered obs to have 16 dimensions, got {obs_filtered.shape[1]}"
        )

        # Extract x and z components from actions
        act_trimmed = act[:, [0, 2]]

        # Apply sim2real scaling transformations
        obs_scaled = scale_sim_obs(obs_filtered)
        act_scaled = scale_sim_action(act_trimmed)

        data = {
            'obs': obs_scaled,
            'action': act_scaled,
        }

        # Add domain encoding for sim data: [1, 0]
        if self.use_domain_encoding:
            domain_encoding = np.zeros(self.domain_encoding_dim, dtype=np.float32)
            domain_encoding[0] = 1.0  # First position is 1 for sim
            data['domain_encoding'] = domain_encoding

        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training sample with optional upsampling and noise augmentation.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing torch tensors for 'obs', 'action', and optionally 'domain_encoding'
        """
        # Sample sequence from replay buffer
        raw_sample = self.sampler.sample_sequence(idx)

        # Subsample if using upsampled data
        if self.upsampled:
            raw_sample = self.sampler.sample_sequence(idx)
            # Subsample every `upsample_multiplier` frame to restore original timing
            for k in [self.obs_key, self.action_key]:
                raw_sample[k] = raw_sample[k][::self.upsample_multiplier]
        else:
            raw_sample = self.sampler.sample_sequence(idx)

        # Process and transform data
        data = self._sample_to_data(raw_sample)
        
        # Add noise augmentation
        data = add_noise(data)
        
        # Apply force binning (if enabled) - AFTER noise
        data = self.apply_force_binning(data)
        
        # Convert to torch tensors
        torch_data = dict_apply(data, torch.from_numpy)
        
        # Ensure domain_encoding is float32
        if self.use_domain_encoding:
            torch_data['domain_encoding'] = torch_data['domain_encoding'].float()

        return torch_data