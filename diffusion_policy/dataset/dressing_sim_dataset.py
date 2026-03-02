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
from diffusion_policy.dataset.util import fix_small_variance_normalizer


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
            use_domain_encoding=False,
            domain_encoding_dim=2,
            seed=42,
            val_ratio=0.0,
            upsampled=True,
            upsample_multiplier=5,
            duplicate_for_pretraining=False
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
        self.duplicate_for_pretraining = duplicate_for_pretraining

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

        data = self._sample_to_data(self.replay_buffer)

        if self.duplicate_for_pretraining:
            data['obs'] = np.concatenate([data['obs'], data['obs']], axis=-1)

        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)

        fix_small_variance_normalizer(normalizer, key='obs')
        fix_small_variance_normalizer(normalizer, key='action')

        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer[self.action_key])

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        obs = sample[self.obs_key]  # shape [T, 38]
        act = sample[self.action_key]  # shape [T, D_a]

        assert obs.ndim == 2, f"Expected obs to be 2D, got {obs.ndim}D"
        assert obs.shape[1] == 38, f"Expected obs to have 38 dimensions, got {obs.shape[1]}"

        obs_filtered = filter_sim_obs(obs)
        assert obs_filtered.shape[1] == 16, (
            f"Expected filtered obs to have 16 dimensions, got {obs_filtered.shape[1]}"
        )

        act_trimmed = act[:, [0, 2]]

        obs_scaled = scale_sim_obs(obs_filtered)
        act_scaled = scale_sim_action(act_trimmed)

        data = {
            'obs': obs_scaled,
            'action': act_scaled,
        }

        if self.use_domain_encoding:
            domain_encoding = np.zeros(self.domain_encoding_dim, dtype=np.float32)
            domain_encoding[0] = 1.0  # First position is 1 for sim
            data['domain_encoding'] = domain_encoding

        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        raw_sample = self.sampler.sample_sequence(idx)

        if self.upsampled:
            for k in [self.obs_key, self.action_key]:
                raw_sample[k] = raw_sample[k][::self.upsample_multiplier]

        data = self._sample_to_data(raw_sample)

        data = add_noise(data, "sim")

        if self.duplicate_for_pretraining:
            data['obs'] = np.concatenate([data['obs'], data['obs']], axis=-1)

        torch_data = dict_apply(data, torch.from_numpy)

        if self.use_domain_encoding:
            torch_data['domain_encoding'] = torch_data['domain_encoding'].float()

        return torch_data
