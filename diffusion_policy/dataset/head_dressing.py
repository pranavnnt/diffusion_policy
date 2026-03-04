import os
import zarr
import torch
import numpy as np

from typing import Dict, List, Optional
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.dressing.head_transforms import (
    filter_head_obs,
    add_noise
)   

from diffusion_policy.dataset.util import fix_small_variance_normalizer

ACTION_THRESHOLD = 1e-3

# Raw observation force indices (from _extract_observation_components in head_transforms.py)
# arm1_force: obs[:, 6:9], arm2_force: obs[:, 15:18]
ARM1_FORCE_SLICE = slice(6, 9)
ARM2_FORCE_SLICE = slice(15, 18)


def filter_zero_force_episodes(
    episode_obs_list: list, episode_actions_list: list
) -> tuple:
    """Filter out episodes where arm1 or arm2 force is all zeros across the entire episode.

    An episode is dropped if all timesteps have arm1_force == 0 OR all timesteps have
    arm2_force == 0 (i.e., the sensor reported no force at all for that robot).

    Args:
        episode_obs_list: List of per-episode obs arrays, each of shape [T, 29]
        episode_actions_list: List of per-episode action arrays, each of shape [T, D_a]

    Returns:
        Tuple of (filtered_obs_list, filtered_actions_list, n_dropped)
    """
    filtered_obs = []
    filtered_actions = []
    n_dropped = 0
    for obs, actions in zip(episode_obs_list, episode_actions_list):
        arm1_force = obs[:, ARM1_FORCE_SLICE]
        arm2_force = obs[:, ARM2_FORCE_SLICE]
        if np.all(arm1_force == 0) or np.all(arm2_force == 0):
            n_dropped += 1
            continue
        filtered_obs.append(obs)
        filtered_actions.append(actions)
    return filtered_obs, filtered_actions, n_dropped


def filter_small_actions(obs: np.ndarray, actions: np.ndarray, threshold: float = ACTION_THRESHOLD) -> tuple:
    """Filter out timesteps where all action values are below threshold.
    
    Keeps only timesteps where at least one action component has absolute value >= threshold.
    Returns both filtered obs and actions to keep them aligned.
    
    Args:
        obs: Observation array of shape [T, D_o]
        actions: Action array of shape [T, D_a]
        threshold: Minimum absolute value for actions (default 1e-2)
        
    Returns:
        Tuple of (filtered_obs, filtered_actions) with small action timesteps removed
    """
    # Create mask: keep timesteps where max absolute action value >= threshold
    mask = np.max(np.abs(actions), axis=-1) >= threshold
    
    filtered_obs = obs[mask]
    filtered_actions = actions[mask]
    
    return filtered_obs, filtered_actions


def load_and_filter_replay_buffer(
    zarr_path: str, 
    obs_key: str, 
    action_key: str,
    threshold: float = ACTION_THRESHOLD
) -> ReplayBuffer:
    """Load zarr data and create a new ReplayBuffer with small actions filtered out.
    
    Args:
        zarr_path: Path to the zarr dataset
        obs_key: Key for observations in the zarr data
        action_key: Key for actions in the zarr data
        threshold: Minimum absolute value for actions (default 1e-2)
        
    Returns:
        ReplayBuffer with filtered data
    """
    # Load original data
    original_buffer = ReplayBuffer.copy_from_path(
        zarr_path=zarr_path,
        store=zarr.MemoryStore(),
        keys=[obs_key, action_key]
    )
    
    # Get episode boundaries
    episode_ends = original_buffer.episode_ends[:]
    all_obs = original_buffer[obs_key][:]
    all_actions = original_buffer[action_key][:]
    
    print(f"Action num before filter: {len(all_actions)}")

    # Collect all episodes
    episode_obs_list = []
    episode_actions_list = []
    start_idx = 0
    for end_idx in episode_ends:
        episode_obs_list.append(all_obs[start_idx:end_idx])
        episode_actions_list.append(all_actions[start_idx:end_idx])
        start_idx = end_idx

    # Filter episodes where either robot's force is all zeros
    episode_obs_list, episode_actions_list, n_zero_force_dropped = filter_zero_force_episodes(
        episode_obs_list, episode_actions_list
    )
    print(f"Episodes dropped due to all-zero force: {n_zero_force_dropped}")

    # Create new replay buffer
    filtered_buffer = ReplayBuffer.create_empty_numpy()

    # Process each episode
    total_before = 0
    total_after = 0
    for episode_obs, episode_actions in zip(episode_obs_list, episode_actions_list):
        total_before += len(episode_obs)

        # Filter small actions
        filtered_obs, filtered_actions = filter_small_actions(
            episode_obs, episode_actions, threshold
        )

        # Only add episode if it has data after filtering
        if len(filtered_obs) > 0:
            filtered_buffer.add_episode(
                data={
                    obs_key: filtered_obs,
                    action_key: filtered_actions
                }
            )
            total_after += len(filtered_obs)

    print(f"Action num after filter: {total_after}")
    
    return filtered_buffer


class HeadDressingDataset(BaseLowdimDataset):
    """Dataset for head dressing task """
    
    def __init__(
        self,
        zarr_configs: List[Dict],
        horizon: int = 1,
        pad_before: int = 0,
        pad_after: int = 0,
        obs_key: str = 'state',
        action_key: str = 'action',
        num_datasets: int = 1,
        include_datasets: Optional[List[str]] = None,
        use_domain_encoding: bool = True,
        domain_encoding_dim: int = 2,
        seed: int = 42
    ):
        super().__init__()
        self._validate_zarr_configs(zarr_configs)

        # Store configuration
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.obs_key = obs_key
        self.action_key = action_key
        self.include_datasets = include_datasets
        self.use_domain_encoding = use_domain_encoding
        self.domain_encoding_dim = domain_encoding_dim

        print("Domain encoding set to: ", self.use_domain_encoding)
        
        # Load in all the zarr datasets
        self.dataset_names = []
        self.replay_buffers = []
        self.train_masks = []
        self.val_masks = []
        self.samplers = []
        self.sample_probabilities = []
        self.zarr_paths = []

        print(f"Datasets included: {self.include_datasets}")

        # Load all zarr datasets
        self._load_datasets(zarr_configs, seed)

        self.num_datasets = len(self.dataset_names)
        assert len(self.dataset_names) == num_datasets, (
            f"num_datasets {num_datasets}, but found {len(self.dataset_names)} included datasets"
        )

        # Normalize sampling probabilities
        self.sample_probabilities = self._normalize_sample_probabilities(
            self.sample_probabilities
        )
        print(f"Sample probabilities: {self.sample_probabilities}")

    def _load_datasets(self, zarr_configs: List[Dict], seed: int) -> None:
        """Load all specified zarr datasets and configure samplers."""
        keys = [self.obs_key, self.action_key]

        for zarr_config in zarr_configs:
            dataset_name = zarr_config['name']
            
            # Skip datasets not in include list
            if dataset_name not in self.include_datasets:
                continue

            self.dataset_names.append(dataset_name)

            # Extract configuration
            zarr_path = zarr_config['path']
            max_train_episodes = zarr_config.get('max_train_episodes', None)
            sampling_weight = zarr_config.get('sampling_weight', None)

            # Load replay buffer with small actions filtered out
            replay_buffer = load_and_filter_replay_buffer(
                zarr_path=zarr_path,
                obs_key=self.obs_key,
                action_key=self.action_key,
                threshold=ACTION_THRESHOLD
            )
            self.replay_buffers.append(replay_buffer)
            n_episodes = replay_buffer.n_episodes

            # Setup train/val masks
            dataset_val_ratio = zarr_config['val_ratio']
            val_mask = get_val_mask(
                n_episodes=n_episodes,
                val_ratio=dataset_val_ratio,
                seed=seed
            )
            train_mask = ~val_mask
            # Note: max_train_episodes is the max number of training episodes,
            # not the total number of train and val episodes!
            train_mask = downsample_mask(
                mask=train_mask,
                max_n=max_train_episodes,
                seed=seed
            )

            self.train_masks.append(train_mask)
            self.val_masks.append(val_mask)

            seq_len = self.horizon


            # Create sampler
            sampler = SequenceSampler(
                replay_buffer=replay_buffer,
                sequence_length=seq_len,
                pad_before=self.pad_before,
                pad_after=self.pad_after,
                episode_mask=train_mask
            )
            self.samplers.append(sampler)

            # Store metadata
            self.sample_probabilities.append(sampling_weight)
            self.zarr_paths.append(zarr_path)

    def get_validation_dataset(self, index: Optional[int] = None) -> 'DressingRealDataset':
        """Create a validation dataset for a specific dataset index.
        
        Args:
            index: Index of dataset to create validation set for. 
                   If None and only one dataset exists, uses index 0.
                   
        Returns:
            Validation dataset instance
        """
        if index is None:
            assert self.num_datasets == 1, (
                "Must specify validation dataset index if multiple datasets"
            )
            index = 0

        # Safely clone the replay buffer with small actions filtered out
        replay_buffer = load_and_filter_replay_buffer(
            zarr_path=self.zarr_paths[index],
            obs_key=self.obs_key,
            action_key=self.action_key,
            threshold=ACTION_THRESHOLD
        )

        # Create a new instance without calling __init__
        val_set = self.__class__.__new__(self.__class__)

        # Manually assign attributes
        val_set.horizon = self.horizon
        val_set.pad_before = self.pad_before
        val_set.pad_after = self.pad_after
        val_set.obs_key = self.obs_key
        val_set.action_key = self.action_key
        val_set.use_domain_encoding = self.use_domain_encoding
        val_set.include_datasets = [self.dataset_names[index]]

        val_set.num_datasets = 1
        val_set.dataset_names = [self.dataset_names[index]]
        val_set.replay_buffers = [replay_buffer]
        val_set.train_masks = [self.train_masks[index]]
        val_set.val_masks = [self.val_masks[index]]
        val_set.zarr_paths = [self.zarr_paths[index]]
        val_set.sample_probabilities = np.array([1.0])
        val_set.domain_encoding_dim = self.domain_encoding_dim  # ADD THIS
        val_set._original_dataset_idx = index  # ADD THIS - track which dataset (0=sim, 1=real)

        # Create validation sampler
        seq_len = self.horizon
        val_set.samplers = [
            SequenceSampler(
                replay_buffer=replay_buffer,
                sequence_length=seq_len,
                pad_before=self.pad_before,
                pad_after=self.pad_after,
                episode_mask=self.val_masks[index]
            )
        ]

        return val_set

    def get_normalizer(self, mode: str = 'limits', **kwargs) -> Optional[LinearNormalizer]:
        """Compute normalizer from simulation or real data.
        
        If simulation data exists, computes normalizer from transformed simulation observations.
        Otherwise, computes normalizer from real data.
        
        Args:
            mode: Normalization mode ('limits' supported)
            
        Returns:
            LinearNormalizer computed from available data
        """
        # Check if we have any sim datasets
        has_sim = any(name.startswith("sim") for name in self.dataset_names)
        
        input_stats = {}
        
        for i, replay_buffer in enumerate(self.replay_buffers):
            # Convert zarr arrays to numpy for fancy indexing
            raw_obs = replay_buffer[self.obs_key][:]
            raw_act = replay_buffer[self.action_key][:]
            
            assert raw_obs.shape[-1] == 29
            
            # Actions are already filtered at load time
            obs_filtered = filter_head_obs(raw_obs)

            data = {
                'obs': obs_filtered,
                'action': raw_act
            }
            normalizer = LinearNormalizer()
            normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
            
            # Update mins and maxes across all datasets
            for key in ['obs', 'action']:
                _max = normalizer[key].params_dict.input_stats.max
                _min = normalizer[key].params_dict.input_stats.min
                
                if key not in input_stats:
                    input_stats[key] = {'max': _max, 'min': _min}
                else:
                    input_stats[key]['max'] = torch.maximum(input_stats[key]['max'], _max)
                    input_stats[key]['min'] = torch.minimum(input_stats[key]['min'], _min)
        
        # Create final normalizer from aggregated stats
        assert len(input_stats) > 0, "No datasets found for computing normalizer"
        normalizer = LinearNormalizer()
        normalizer.fit_from_input_stats(input_stats_dict=input_stats)
        # Fix small variance dimensions
        fix_small_variance_normalizer(normalizer, key='obs')
        fix_small_variance_normalizer(normalizer, key='action')
        return normalizer
        

    def get_sample_probabilities(self) -> np.ndarray:
        """Get normalized sampling probabilities for each dataset."""
        return self.sample_probabilities

    def get_num_datasets(self) -> int:
        """Get total number of loaded datasets."""
        return self.num_datasets

    def get_num_episodes(self, index: Optional[int] = None) -> int:
        """Get number of episodes in dataset(s).
        
        Args:
            index: Specific dataset index, or None for total across all datasets
            
        Returns:
            Number of episodes
        """
        if index is None:
            num_episodes = 0
            for i in range(self.num_datasets):
                num_episodes += self.replay_buffers[i].n_episodes
            return num_episodes
        else:
            return self.replay_buffers[index].n_episodes

    def __len__(self) -> int:
        """Total number of samples across datasets with non-zero sampling probability."""
        length = 0
        for i, sampler in enumerate(self.samplers):
            if self.sample_probabilities[i] > 0:
                length += len(sampler)
        return length

    def _sample_to_data(
        self, 
        sample: Dict[str, np.ndarray], 
        sampler_idx: int
    ) -> Dict[str, np.ndarray]:
        """Convert raw sample to processed data with domain-specific transforms.
        
        Args:
            sample: Raw sample containing observations and actions
            sampler_idx: Index of the sampler/dataset this sample came from
            
        Returns:
            Dictionary with processed 'obs', 'action', and optionally 'domain_encoding'
        """
        # Rename to the standard keys the policy expects
        obs = sample[self.obs_key]  # shape [T, D_o]
        act = sample[self.action_key]  # shape [T, D_a]
        local_dataset_name = self.dataset_names[sampler_idx]

        # Actions are already filtered at load time
        obs_filtered = filter_head_obs(obs)
            
        assert obs_filtered.shape[1] == 23, (
            f"Expected obs dim 23 from {local_dataset_name}, got {obs_filtered.shape[1]}"
        )

        data = {
            'obs': obs_filtered,      # shape [T, D_o]
            'action': act,            # shape [T, D_a] (already filtered at load time)
        }

        if self.use_domain_encoding:
            # Hard-code domain encodings
            if local_dataset_name.startswith("sim"):
                data['domain_encoding'] = np.array([1.0, 0.0], dtype=np.float32)
            else:
                data['domain_encoding'] = np.array([0.0, 1.0], dtype=np.float32)
    
        return data

    def _validate_zarr_configs(self, zarr_configs: List[Dict]) -> None:
        """Validate zarr configuration parameters.
        
        Raises:
            ValueError: If any configuration is invalid
        """
        num_null_sampling_weights = 0
        N = len(zarr_configs)

        for zarr_config in zarr_configs:
            zarr_path = zarr_config['path']
            if not os.path.exists(zarr_path):
                raise ValueError(f"path {zarr_path} does not exist")

            max_train_episodes = zarr_config.get('max_train_episodes', None)
            if max_train_episodes is not None and max_train_episodes <= 0:
                raise ValueError(
                    f"max_train_episodes must be greater than 0, got {max_train_episodes}"
                )

            sampling_weight = zarr_config.get('sampling_weight', None)
            if sampling_weight is None:
                num_null_sampling_weights += 1
            elif sampling_weight < 0:
                raise ValueError(
                    f"sampling_weight must be greater than or equal to 0, got {sampling_weight}"
                )

        if num_null_sampling_weights not in [0, N]:
            raise ValueError("Either all or none of the zarr_configs must have a sampling_weight")

    def _normalize_sample_probabilities(
        self, 
        sample_probabilities: List[float]
    ) -> np.ndarray:
        """Normalize sampling probabilities to sum to 1.
        
        Args:
            sample_probabilities: List of sampling weights
            
        Returns:
            Normalized probability array
        """
        total = np.sum(sample_probabilities)
        assert total > 0, "Sum of sampling weights must be greater than 0"
        return sample_probabilities / total
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training sample with transformations and noise augmentation.
        
        Args:
            idx: Sample index (ignored if multiple datasets, uses random sampling)
            
        Returns:
            Dictionary containing torch tensors for 'obs', 'action', 
            and optionally 'domain_encoding'
        """
        if self.num_datasets == 1:
            sampler_idx = 0
            local_idx = idx
        else:
            sampler_idx = np.random.choice(self.num_datasets, p=self.sample_probabilities)
            local_idx = np.random.randint(len(self.samplers[sampler_idx]))

        sampler = self.samplers[sampler_idx]
        sample = sampler.sample_sequence(local_idx)

        # determine
        dataset_name = self.dataset_names[sampler_idx]

        # Use original dataset index if this is a validation set
        data = self._sample_to_data(sample, sampler_idx)
        data = add_noise(data, dataset_name)
        torch_data = dict_apply(data, torch.from_numpy)

        if self.use_domain_encoding:
            torch_data['domain_encoding'] = torch_data['domain_encoding'].float()

        return torch_data