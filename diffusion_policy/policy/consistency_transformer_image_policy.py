from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply

from diffusion_policy.model.consistency.karras_diffusion import KarrasDenoiserForTransformer, karras_sample_transformer
from diffusion_policy.model.consistency.sampler import create_named_schedule_sampler
from diffusion_policy.model.consistency.scripts_util import create_ema_and_scales_fn
from diffusion_policy.model.consistency.nn import update_ema
from diffusion_policy.model.consistency.fp16_utils import master_params_to_model_params, make_master_params, get_param_groups_and_shapes

import functools
import copy

class ConsistencyTransformerImagePolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler,
            ema_scale,
            sample,
            obs_encoder: MultiImageObsEncoder,
            # task params
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            # arch
            n_layer=8,
            n_cond_layers=0,
            n_head=4,
            n_emb=256,
            p_drop_emb=0.0,
            p_drop_attn=0.3,
            causal_attn=True,
            time_as_cond=True,
            obs_as_cond=True,
            pred_action_steps_only=False,
            # parameters passed to step
            **kwargs):
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        # get feature dim
        obs_feature_dim = obs_encoder.output_shape()[0]

        # create diffusion model
        input_dim = action_dim if obs_as_cond else (obs_feature_dim + action_dim)
        output_dim = input_dim
        cond_dim = obs_feature_dim if obs_as_cond else 0

        model = TransformerForDiffusion(
            input_dim=input_dim,
            output_dim=output_dim,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            cond_dim=cond_dim,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
            causal_attn=causal_attn,
            time_as_cond=time_as_cond,
            obs_as_cond=obs_as_cond,
            n_cond_layers=n_cond_layers
        )
        
        # ema and scale
        self.ema_scale_fn = create_ema_and_scales_fn(
            target_ema_mode=ema_scale.target_ema_mode,
            start_ema=ema_scale.start_ema,
            scale_mode=ema_scale.scale_mode,
            start_scales=ema_scale.start_scales,
            end_scales=ema_scale.end_scales,
            total_steps=ema_scale.total_training_steps,
            distill_steps_per_iter=ema_scale.distill_steps_per_iter,
        )

        # model and target model
        self.model = model
        self.param_groups_and_shapes = get_param_groups_and_shapes(
            self.model.named_parameters()
        )
        self.master_params = make_master_params(
            self.param_groups_and_shapes
        )

        self.target_model = copy.deepcopy(self.model)
        self.target_model.requires_grad_(False)
        self.target_model.train()

        self.target_model_param_groups_and_shapes = get_param_groups_and_shapes(
            self.target_model.named_parameters()
        )
        self.target_model_master_params = make_master_params(
            self.target_model_param_groups_and_shapes
        )

        # CM scheduler
        self.diffusion = KarrasDenoiserForTransformer(
            sigma_data=noise_scheduler.sigma_data,
            sigma_max=noise_scheduler.sigma_max,
            sigma_min=noise_scheduler.sigma_min,
            distillation=noise_scheduler.distillation,
            weight_schedule=noise_scheduler.weight_schedule,
            noise_schedule=noise_scheduler.noise_schedule,
            loss_norm=noise_scheduler.loss_norm,
        )
        self.schedule_sampler = create_named_schedule_sampler(
            noise_scheduler.schedule_sampler,
            self.diffusion
        )

        self.obs_encoder = obs_encoder
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_cond) else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_cond = obs_as_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.kwargs = kwargs

        # CM-related parameters
        self.training_mode = ema_scale.training_mode

        self.sampler = sample.sampler
        self.generator = sample.generator
        self.clip_denoised = sample.clip_denoised
        self.sigma_min = noise_scheduler.sigma_min
        self.sigma_max = noise_scheduler.sigma_max
        self.steps = sample.steps

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
    

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        cond = None
        cond_data = None
        cond_mask = None
        if self.obs_as_cond:
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, To, Do
            cond = nobs_features.reshape(B, To, -1)
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, To, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            shape = (B, T, Da+Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = karras_sample_transformer(
            self.diffusion,
            self.model,
            (B, T, Da),
            cond_data,
            cond_mask,
            steps=self.steps,
            clip_denoised=self.clip_denoised,
            cond=cond,
            device=self.device,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            sampler=self.sampler,
            generator=None,
        )
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
            self, 
            transformer_weight_decay: float, 
            obs_encoder_weight_decay: float,
            learning_rate: float, 
            betas: Tuple[float, float]
        ) -> torch.optim.Optimizer:
        optim_groups = self.model.get_optim_groups(
            weight_decay=transformer_weight_decay)
        optim_groups.append({
            "params": self.obs_encoder.parameters(),
            "weight_decay": obs_encoder_weight_decay
        })
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer

    def compute_loss(self, batch, global_step):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]
        To = self.n_obs_steps

        # handle different ways of passing observation
        cond = None
        trajectory = nactions
        if self.obs_as_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            cond = nobs_features.reshape(batch_size, To, -1)
            if self.pred_action_steps_only:
                start = To - 1
                end = start + self.n_action_steps
                trajectory = nactions[:,start:end]
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            trajectory = torch.cat([nactions, nobs_features], dim=-1).detach()

        # generate impainting mask
        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        # importance and scales
        _, weights = self.schedule_sampler.sample(trajectory.shape[0], self.device)
        _, num_scales = self.ema_scale_fn(global_step)

        # loss type and computation
        if self.training_mode == "consistency_training":
            compute_losses = functools.partial(
                self.diffusion.consistency_losses,
                self.model,
                trajectory,
                num_scales,
                cond=cond,
                target_model=self.target_model,
            )
        else:
            raise ValueError(f"Warning training mode {self.training_mode}")

        losses = compute_losses()
        loss = (losses["loss"] * weights).mean()

        return loss

    def update_target_ema(self, global_step):
        self.master_params = make_master_params(
            self.param_groups_and_shapes
        )
        self.target_model_master_params = make_master_params(
            self.target_model_param_groups_and_shapes
        )

        target_ema, _ = self.ema_scale_fn(global_step)
        with torch.no_grad():
            update_ema(
                self.target_model_master_params,
                self.master_params,
                rate = target_ema,
            )
            master_params_to_model_params(
                self.target_model_param_groups_and_shapes,
                self.target_model_master_params,
            )
