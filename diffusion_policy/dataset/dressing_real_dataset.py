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
from diffusion_policy.dressing.sim2real_transforms import (
    filter_sim_obs,
    scale_sim_obs,
    scale_sim_action,
    add_noise
)


class DressingRealDataset(BaseLowdimDataset):
    """Dataset for dressing task supporting both simulation and real-world data.
    
    This dataset can load multiple zarr datasets (sim and/or real) and sample from them
    with configurable probabilities. It handles domain-specific transformations and
    provides domain encodings for domain adaptation.
    """
    
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
        self.upsample_multipliers = []

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

            # Load replay buffer
            replay_buffer = ReplayBuffer.copy_from_path(
                zarr_path=zarr_path,
                store=zarr.MemoryStore(),
                keys=keys
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

            # Handle upsampling configuration
            assert 'upsampled' in zarr_config, (
                "Must specify if dataset is upsampled or not, in zarr_config"
            )
            assert 'upsample_multiplier' in zarr_config, (
                "Must specify upsample_multiplier in zarr_config"
            )
            upsampled = zarr_config['upsampled']
            upsample_multiplier = zarr_config['upsample_multiplier'] if upsampled else 1
            if upsampled:
                assert upsample_multiplier > 1, (
                    "upsample_multiplier must be greater than 1 for upsampled datasets"
                )
            self.upsample_multipliers.append(upsample_multiplier)

            # Get sequence length
            seq_len = self.horizon * upsample_multiplier if upsampled else self.horizon

            # Create sampler
            sampler = SequenceSampler(
                replay_buffer=replay_buffer,
                sequence_length=seq_len,
                pad_before=self.pad_before,
                pad_after=self.pad_after,
                episode_mask=train_mask,
                stride=upsample_multiplier
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

        # Safely clone the replay buffer
        replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path=self.zarr_paths[index],
            store=zarr.MemoryStore(),
            keys=[self.obs_key, self.action_key]
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
        val_set.upsample_multipliers = [self.upsample_multipliers[index]]
        val_set.sample_probabilities = np.array([1.0])
        val_set.domain_encoding_dim = self.domain_encoding_dim  # ADD THIS
        val_set._original_dataset_idx = index  # ADD THIS - track which dataset (0=sim, 1=real)

        # Create validation sampler
        seq_len = self.horizon * self.upsample_multipliers[index]
        val_set.samplers = [
            SequenceSampler(
                replay_buffer=replay_buffer,
                sequence_length=seq_len,
                pad_before=self.pad_before,
                pad_after=self.pad_after,
                episode_mask=self.val_masks[index],
                stride=self.upsample_multipliers[index]
            )
        ]

        return val_set

    def get_normalizer(self, mode: str = 'limits', **kwargs) -> Optional[LinearNormalizer]:
        """Compute normalizer from simulation data.
        
        For fine-tuning with real data only, returns None (normalizer should be 
        loaded from pretrained checkpoint). For datasets with simulation data,
        computes normalizer statistics from transformed simulation observations.
        
        Args:
            mode: Normalization mode ('limits' supported)
            
        Returns:
            LinearNormalizer if sim data exists, None otherwise
        """
        # Compute mins and maxes
        assert mode == 'limits', "Only supports limits mode"
        input_stats = {}

        # Check if we have any sim datasets
        has_sim = any(name.startswith("sim") for name in self.dataset_names)

        if not has_sim:
            # For finetuning: Return None - normalizer will be loaded from checkpoint
            print("No sim datasets found. Normalizer should be loaded from pretrained checkpoint.")
            return None

        # Original sim-based normalization code
        for i, replay_buffer in enumerate(self.replay_buffers):
            
            # Use only sim data for normalization
            if self.dataset_names[i].startswith("sim"):
                raw_obs = replay_buffer[self.obs_key]
                raw_act = replay_buffer[self.action_key]

                assert raw_obs.shape[-1] == 37, f"Shape of raw_obs is {raw_obs.shape}, expected last dim to be 37"

                # Filter & scale ALL sim obs BEFORE computing normals
                obs_filt = filter_sim_obs(raw_obs)
                obs_scaled = scale_sim_obs(obs_filt)

                # Trim + scale sim actions BEFORE computing normals
                raw_act = raw_act[:]
                act_scaled = scale_sim_action(raw_act)

                data = {
                    'obs': obs_scaled,
                    'action': act_scaled
                }
                normalizer = LinearNormalizer()
                normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)

                # Update mins and maxes
                for key in ['obs', 'action']:
                    _max = normalizer[key].params_dict.input_stats.max
                    _min = normalizer[key].params_dict.input_stats.min

                    if key not in input_stats:
                        input_stats[key] = {'max': _max, 'min': _min}
                    else:
                        input_stats[key]['max'] = torch.maximum(input_stats[key]['max'], _max)
                        input_stats[key]['min'] = torch.minimum(input_stats[key]['min'], _min)

        # Create normalizer
        # Normalizer is a PyTorch parameter dict containing normalizers for all the keys
        assert len(input_stats) > 0, "No simulation datasets found for computing normalizer"
        normalizer = LinearNormalizer()
        normalizer.fit_from_input_stats(input_stats_dict=input_stats)
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
        """Total number of samples across all datasets."""
        length = 0
        for sampler in self.samplers:
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

        obs_scaled = obs_trimmed = obs
        act_scaled = act_trimmed = act
        
        local_dataset_name = self.dataset_names[sampler_idx]

        if local_dataset_name.startswith("sim"):
            # Simulation data: apply full transformation pipeline
            obs_trimmed = filter_sim_obs(obs)
            obs_scaled = scale_sim_obs(obs_trimmed)
            act_scaled = scale_sim_action(act_trimmed)
        else:
            # Real world data: already in correct format
            pass
            
        assert obs_scaled.shape[1] == 36, (
            f"Expected obs dim 36, got {obs_scaled.shape[1]}"
        )

        data = {
            'obs': obs_scaled,      # shape [T, D_o]
            'action': act_scaled,   # shape [T, D_a]
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