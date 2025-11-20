import os
import copy
import zarr
import torch
import numpy as np

from typing import Dict
from torchvision import transforms

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer 
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.dressing.sim_transforms import filter_sim_obs, scale_sim_obs, scale_sim_action

class DressingRealLowdimDataset(BaseLowdimDataset):
    def __init__(self, 
            zarr_configs,
            horizon=1,
            pad_before=0,
            pad_after=0,
            obs_key='state',
            action_key='action',
            num_datasets=1,
            include_datasets=None,
            use_domain_encoding=True, 
            seed=42):

        super().__init__()
        self._validate_zarr_configs(zarr_configs)

        # Load other variables
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.obs_key = obs_key
        self.action_key = action_key
        self.include_datasets = include_datasets
        self.use_domain_encoding = use_domain_encoding
        
        # Load in all the zarr datasets
        self.num_datasets = len(zarr_configs)
        self.dataset_names = []
        self.replay_buffers = []
        self.train_masks = []
        self.val_masks = []
        self.samplers = []
        self.sample_probabilities = np.zeros(len(zarr_configs))
        self.zarr_paths = []
        self.upsample_multipliers = []

        for i, zarr_config in enumerate(zarr_configs):
            
            # extract name
            dataset_name = zarr_config['name']
            self.dataset_names.append(dataset_name)

            if dataset_name not in self.include_datasets:
                continue
            
            # extract config info
            zarr_path = zarr_config['path']
            max_train_episodes = zarr_config.get('max_train_episodes', None)
            sampling_weight = zarr_config.get('sampling_weight', None)
            
            keys = [obs_key, action_key]

            # Set up replay buffer
            self.replay_buffers.append(ReplayBuffer.copy_from_path(
                    zarr_path=zarr_path, 
                    store=zarr.MemoryStore(),
                    keys=keys
                )
            )
            n_episodes = self.replay_buffers[-1].n_episodes

            # Set up masks
            dataset_val_ratio = zarr_config['val_ratio']
            val_mask = get_val_mask(
                n_episodes=n_episodes, 
                val_ratio=dataset_val_ratio,
                seed=seed)
            train_mask = ~val_mask
            # Note max_train_episodes is the max number of training episodes
            # not the total number of train and val episodes!
            train_mask = downsample_mask(
                mask=train_mask, 
                max_n=max_train_episodes, 
                seed=seed)
            
            self.train_masks.append(train_mask)
            self.val_masks.append(val_mask)

            # Get upsample info
            assert 'upsampled' in zarr_config, "Must specify if dataset is upsampled or not, in zarr_config"
            assert 'upsample_multiplier' in zarr_config, "Must specify upsample_multiplier in zarr_config"
            upsampled = zarr_config['upsampled']
            upsample_multiplier = zarr_config['upsample_multiplier'] if upsampled else 1
            if upsampled:
                assert upsample_multiplier > 1, "upsample_multiplier must be greater than 1 for upsampled datasets"
            self.upsample_multipliers.append(upsample_multiplier)
            
            # get sequence length
            seq_len = self.horizon * upsample_multiplier if upsampled else self.horizon

            # Set up sampler
            self.samplers.append(
                SequenceSampler(
                    replay_buffer=self.replay_buffers[-1], 
                    sequence_length=seq_len,
                    pad_before=pad_before, 
                    pad_after=pad_after,
                    episode_mask=train_mask
                )
            )
            
            # Set up sample probabilities and zarr paths
            if sampling_weight is not None:
                self.sample_probabilities[i] = sampling_weight
            else:
                self.sample_probabilities[i] = np.sum(train_mask)
            self.zarr_paths.append(zarr_path)

        assert len(self.dataset_names) == num_datasets, f"num_datasets {num_datasets}, but found {len(self.dataset_names)} included datasets"

        # Normalize sample_probabilities
        self.sample_probabilities = self._normalize_sample_probabilities(self.sample_probabilities)
        print("Sample probabilities:", self.sample_probabilities)

    def get_validation_dataset(self, index=None):
        # Create validation dataset
        val_set = copy.copy(self)

        if index == None:
            assert self.num_datasets == 1, "Must specify validation dataset index if multiple datasets"
            index = 0
        else:
            val_set.replay_buffers = [self.replay_buffers[index]]
            val_set.train_masks = [self.train_masks[index]]
            val_set.val_masks = [self.val_masks[index]]
            val_set.zarr_paths = [self.zarr_paths[index]]
        val_set.num_datasets = 1
        val_set.sample_probabilities = np.array([1.0])

        # Set one hot encoding
        val_set.domain_encoding = np.zeros(self.num_datasets).astype(np.float32)
        val_set.domain_encoding[index] = 1

        upsample_multiplier = self.upsample_multipliers[index]
        
        # get sequence length
        seq_len = self.horizon * upsample_multiplier

        val_set.samplers = [SequenceSampler(
            replay_buffer=self.replay_buffers[index], 
            sequence_length=seq_len,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=self.val_masks[index]
        )]
        
        return val_set
    
    def get_normalizer(self, mode='limits', **kwargs):
        # compute mins and maxes
        assert mode == 'limits', "Only supports limits mode"
        input_stats = {}
        for i, replay_buffer in enumerate(self.replay_buffers):
            
            # Use only sim data for normalization!!!
            if self.dataset_names[i].startswith("sim"):
                raw_obs = replay_buffer[self.obs_key]
                raw_act = replay_buffer[self.action_key]

                assert raw_obs.shape[-1] == 37, f"Shape of raw_obs is {raw_obs.shape}, expected last dim to be 37"
                # Filter & scale ALL sim obs BEFORE computing normals
                obs_filt = filter_sim_obs(raw_obs)
                obs_scaled = scale_sim_obs(obs_filt)

                # Trim + scale sim actions BEFORE computing normals
                raw_act = raw_act[:]
                act_trim = raw_act[:, [0, 2]]
                act_scaled = scale_sim_action(act_trim)

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

    def get_sample_probabilities(self):
        return self.sample_probabilities
    
    def get_num_datasets(self):
        return self.num_datasets
    
    def get_num_episodes(self, index=None):
        if index == None:
            num_episodes = 0
            for i in range(self.num_datasets):
                num_episodes += self.replay_buffers[i].n_episodes
            return num_episodes
        else:
            return self.replay_buffers[index].n_episodes

    def __len__(self) -> int:
        length = 0
        for sampler in self.samplers:
            length += len(sampler)
        return length

    def _sample_to_data(self, sample, sampler_idx):
        
        # Rename to the standard keys the policy expects:

        obs = sample[self.obs_key]        # shape [T, D_o]
        act = sample[self.action_key]     # shape [T, D_a]

        obs_scaled = obs_trimmed = obs
        act_scaled = act_trimmed = act[:, [0, 2]]
        if self.dataset_names[sampler_idx].startswith("sim"):
            # print(f"Using simulation data, dataset: {self.dataset_names[sampler_idx]}")
            obs_trimmed = filter_sim_obs(obs)
            obs_scaled  = scale_sim_obs(obs_trimmed)
            act_scaled  = scale_sim_action(act_trimmed)
            # print(f"obs_scaled shape: {obs_scaled.shape}, act_scaled shape: {act_scaled.shape}")
        else:
            # real world data
            # print(f"Using real world data, dataset: {self.dataset_names[sampler_idx]}")
            pass
        assert obs_scaled.shape[1] == 16, f"Expected obs dim 16, got {obs_scaled.shape[1]}"
        
        data = {
            'obs': obs_scaled,               # shape [T, D_o]
            'action': act_scaled,            # shape [T, D_a]
        }

        if self.use_domain_encoding:
            # domain encoding is one-hot
            data['domain_encoding'] = np.zeros(self.num_datasets).astype(np.float32)
            data['domain_encoding'][sampler_idx] = 1            
            
        return data
    
    def _validate_zarr_configs(self, zarr_configs):
        num_null_sampling_weights = 0
        N = len(zarr_configs)

        for zarr_config in zarr_configs:
            zarr_path = zarr_config['path']
            if not os.path.exists(zarr_path):
                raise ValueError(f"path {zarr_path} does not exist")
            
            max_train_episodes = zarr_config.get('max_train_episodes', None)
            if max_train_episodes is not None and max_train_episodes <= 0:
                raise ValueError(f"max_train_episodes must be greater than 0, got {max_train_episodes}")
            
            sampling_weight = zarr_config.get('sampling_weight', None)
            if sampling_weight is None:
                num_null_sampling_weights += 1
            elif sampling_weight < 0:
                raise ValueError(f"sampling_weight must be greater than or equal to 0, got {sampling_weight}")
        
        if num_null_sampling_weights not in [0, N]:
            raise ValueError("Either all or none of the zarr_configs must have a sampling_weight")
    
    def _normalize_sample_probabilities(self, sample_probabilities):
        total = np.sum(sample_probabilities)
        assert total > 0, "Sum of sampling weights must be greater than 0"
        return sample_probabilities / total
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # To sample a sequence, first sample a dataset,
        # then sample a sequence from that dataset
        # Note that this implementation does not guarantee that each unique
        # sequence is sampled on every epoch!
        
        # Get sample
        if self.num_datasets == 1:
            sampler_idx = 0
            sampler = self.samplers[sampler_idx]
            raw_sample = sampler.sample_sequence(idx)
        else:
            sampler_idx = np.random.choice(self.num_datasets, p=self.sample_probabilities)
            sampler = self.samplers[sampler_idx]
            raw_sample = sampler.sample_sequence(idx)

        mult = self.upsample_multipliers[sampler_idx]
        if mult > 1:
            for k in raw_sample.keys():
                raw_sample[k] = raw_sample[k][::mult]
        sample = raw_sample
        
        # Process sample
        data = self._sample_to_data(sample, sampler_idx)
        torch_data = dict_apply(data, torch.from_numpy)
        torch_data['domain_encoding'] = torch_data['domain_encoding'].float()
        return torch_data
