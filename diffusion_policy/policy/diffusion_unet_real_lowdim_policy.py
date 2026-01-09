"""
Policy for dressing task with dual observations.
Handles separate encoders for state and distilled_features.
"""
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator


def create_mlp(input_dim: int, output_dim: int, hidden_dims: list, activation: str = 'relu') -> nn.Module:
    """Create a simple MLP."""
    layers = []
    dims = [input_dim] + hidden_dims + [output_dim]
    
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        if i < len(dims) - 2:  # No activation after last layer
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'elu':
                layers.append(nn.ELU())
    
    return nn.Sequential(*layers)


class DiffusionUnetRealLowdimPolicy(BaseLowdimPolicy):
    """
    Handles dual observations:
    - state: low-dim robot state
    - distilled_features: high-dim visual features
    
    Each observation type gets its own encoder, then concatenated
    before passing to the diffusion model.
    """
    
    def __init__(
        self,
        model: ConditionalUnet1D,
        noise_scheduler: DDPMScheduler,
        horizon: int,
        obs_dim: int,  # Not used directly, kept for compatibility
        action_dim: int,
        n_action_steps: int,
        n_obs_steps: int,
        state_dim: int,
        distilled_features_dim: int,
        state_encoder_hidden_dims: list = [128],
        distilled_encoder_hidden_dims: list = [128],
        encoder_output_dim: int = 64,
        num_inference_steps=None,
        obs_as_local_cond=False,
        obs_as_global_cond=False,
        pred_action_steps_only=False,
        oa_step_convention=False,
        **kwargs
    ):
        super().__init__()
        
        assert not (obs_as_local_cond and obs_as_global_cond)
        if pred_action_steps_only:
            assert obs_as_global_cond
        
        # Store dimensions
        self.state_dim = state_dim
        self.distilled_features_dim = distilled_features_dim
        self.encoder_output_dim = encoder_output_dim
        self.combined_obs_dim = 2 * encoder_output_dim  # Combined encoded dimension
        
        # Create encoders
        self.state_encoder = create_mlp(
            input_dim=state_dim,
            output_dim=encoder_output_dim,
            hidden_dims=state_encoder_hidden_dims,
            activation='relu'
        )
        
        self.distilled_encoder = create_mlp(
            input_dim=distilled_features_dim,
            output_dim=encoder_output_dim,
            hidden_dims=distilled_encoder_hidden_dims,
            activation='relu'
        )
        
        # Store model and scheduler
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_local_cond or obs_as_global_cond) else self.combined_obs_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_dim = self.combined_obs_dim  # Use combined dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_local_cond = obs_as_local_cond
        self.obs_as_global_cond = obs_as_global_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.oa_step_convention = oa_step_convention
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
    
    def encode_observations(self, state: torch.Tensor, distilled_features: torch.Tensor) -> torch.Tensor:
        """
        Encode state and distilled_features separately, then concatenate.
        
        Args:
            state: [B, T, state_dim] or [B, state_dim]
            distilled_features: [B, T, distilled_dim] or [B, distilled_dim]
            
        Returns:
            combined: [B, T, combined_obs_dim] or [B, combined_obs_dim]
        """
        state_enc = self.state_encoder(state)
        distilled_enc = self.distilled_encoder(distilled_features)
        combined = torch.cat([state_enc, distilled_enc], dim=-1)
        return combined
    
    # ========= inference  ============
    def conditional_sample(self, condition_data, condition_mask, local_cond=None, global_cond=None, generator=None, **kwargs):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            trajectory[condition_mask] = condition_data[condition_mask]
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
            ).prev_sample
        
        trajectory[condition_mask] = condition_data[condition_mask]        
        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Predict actions from dual observations.
        
        Args:
            obs_dict: Dictionary containing:
                - 'state': state observations [B, T, state_dim]
                - 'distilled_features': visual features [B, T, distilled_dim]
        
        Returns:
            action_dict: Dictionary with predicted actions
        """
        assert 'state' in obs_dict
        assert 'distilled_features' in obs_dict
        
        # Normalize state and distilled features separately
        nstate = self.normalizer['state'].normalize(obs_dict['state'])
        ndistilled = self.normalizer['distilled_features'].normalize(obs_dict['distilled_features'])
        
        # Encode observations
        nobs = self.encode_observations(nstate, ndistilled)
        
        B, _, Do = nobs.shape
        To = self.n_obs_steps
        assert Do == self.combined_obs_dim
        T = self.horizon
        Da = self.action_dim

        device = self.device
        dtype = self.dtype

        # Handle different ways of passing observation
        local_cond = None
        global_cond = None
        
        if self.obs_as_local_cond:
            local_cond = torch.zeros(size=(B,T,Do), device=device, dtype=dtype)
            local_cond[:,:To] = nobs[:,:To]
            shape = (B, T, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        elif self.obs_as_global_cond:
            global_cond = nobs[:,:To].reshape(nobs.shape[0], -1)
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            shape = (B, T, Da+Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs[:,:To]
            cond_mask[:,:To,Da:] = True

        # Run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        # Unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # Get action
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To
            if self.oa_step_convention:
                start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        
        if not (self.obs_as_local_cond or self.obs_as_global_cond):
            nobs_pred = nsample[...,Da:]
            obs_pred = self.normalizer['obs'].unnormalize(nobs_pred)
            action_obs_pred = obs_pred[:,start:end]
            result['action_obs_pred'] = action_obs_pred
            result['obs_pred'] = obs_pred
            
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        """
        Compute diffusion loss with dual observations.
        
        Args:
            batch: Dictionary containing:
                - 'state': state observations [B, T, state_dim]
                - 'distilled_features': visual features [B, T, distilled_dim]
                - 'action': actions [B, T, action_dim]
        
        Returns:
            loss: Scalar loss value
        """
        # Normalize state and distilled features separately
        keys_to_normalize = ['state', 'distilled_features', 'action']
        nbatch = {}
        for k in keys_to_normalize:
            if k in batch:
                try:
                    nbatch[k] = self.normalizer[k].normalize(batch[k])
                except KeyError:
                    # Key not in normalizer, use raw value
                    nbatch[k] = batch[k]
        
        # Encode observations
        nobs = self.encode_observations(nbatch['state'], nbatch['distilled_features'])
        action = nbatch['action']

        # Handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = action
        
        if self.obs_as_local_cond:
            local_cond = nobs
            local_cond[:,self.n_obs_steps:,:] = 0
        elif self.obs_as_global_cond:
            global_cond = nobs[:,:self.n_obs_steps,:].reshape(nobs.shape[0], -1)
            
            if self.pred_action_steps_only:
                To = self.n_obs_steps
                start = To
                if self.oa_step_convention:
                    start = To - 1
                end = start + self.n_action_steps
                trajectory = action[:,start:end]
        else:
            trajectory = torch.cat([action, nobs], dim=-1)

        # Generate impainting mask
        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        
        # Sample random timestep
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        
        # Add noise (forward diffusion)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # Compute loss mask
        loss_mask = ~condition_mask

        # Apply conditioning
        noisy_trajectory[condition_mask] = trajectory[condition_mask]
        
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        
        return loss