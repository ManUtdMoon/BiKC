from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.trajectory_base_policy import TrajectoryBasePolicy
from diffusion_policy.model.diffusion.goal_cond_transformer_for_diffusion import GoalCondTransformerForDiffusion
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply


class TrajectoryTransformerPolicy(TrajectoryBasePolicy):
    def __init__(self,
        shape_meta: Dict,
        noise_scheduler: DDIMScheduler,
        obs_encoder: MultiImageObsEncoder,
        # task params
        horizon,
        n_action_steps, 
        n_obs_steps,
        num_inference_steps=None,
        # arch
        n_layer=4,
        n_head=4,
        n_emb=256,
        p_drop_attn=0.1,
        causal_attn=True,
        obs_as_cond=True,
        pred_action_steps_only=False,
        # params for diffusion step
        **kwargs
    ):
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
        obs_dim = obs_feature_dim if obs_as_cond else 0
        model = GoalCondTransformerForDiffusion(
            input_dim=input_dim,
            output_dim=output_dim,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            obs_dim=obs_dim,
            n_layer=n_layer,
            n_head=n_head,
            d_embedding=n_emb,
            p_drop=p_drop_attn,
            causal_attn=causal_attn,
            obs_as_cond=obs_as_cond,
        )

        # save class variables
        self.obs_encoder = obs_encoder
        self.model = model
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

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    # ========== inference ==========
    def conditional_sample(self,
        condition_data: torch.Tensor,
        condition_mask: torch.Tensor,
        cond=None,
        goal=None,
        generator=None,
        # params for diffusion step
        **kwargs
    ) -> torch.Tensor:
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator
        )

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, cond, goal)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
            ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory


    def predict_trajectory(self,
        obs_dict: Dict[str, torch.Tensor],
        next_keypose: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        @params:
            obs_dict: {key: tensor (B,To,...)}
            next_keypose: B,Da
        @return:
            trajectory: B,Ta,Da
            trajectory_pred: B,T/H,Da

        """
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        nnext_keypose = self.normalizer['next_keypose'].normalize(next_keypose)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        ### build input
        device = self.device
        dtype = self.dtype

        # obs and goal as condition
        this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        # reshape back to B, To, Do
        cond = nobs_features.reshape(B, To, -1)
        goal = nnext_keypose

        # prepare condition data and mask, actually we don't need this
        shape = (B, T, Da)
        if self.pred_action_steps_only:
            shape = (B, self.n_action_steps, Da)
        cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            cond=cond,
            goal=goal,
            **self.kwargs
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
            "action": action,
            "action_pred": action_pred
        }
        return result

    # ========== training ==========
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(self, 
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

    def compute_loss(self,
        batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        batch: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]
            obs_dict:
                str: B,To,*
            next_keypose: B,Da
            action: B,Ta,Da
        return: torch.Tensor
        """
        ### normalize and get constants
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        nnext_keypose = self.normalizer['next_keypose'].normalize(batch['next_keypose'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]
        To = self.n_obs_steps

        ### prepare obs cond and goal
        # reshape (B,T,...) to (B*T,...)
        this_nobs = dict_apply(nobs, 
            lambda x: x[:,:To,...].reshape(-1,*x.shape[2:])
        )
        nobs_features = self.obs_encoder(this_nobs)
        # reshape back to (B,T,Do)
        cond = nobs_features.reshape(batch_size, To, -1)
        goal = nnext_keypose

        ### generate impainting mask
        trajectory = nactions
        if self.pred_action_steps_only:
            start = To - 1
            end = start + self.n_action_steps
            trajectory = nactions[:,start:end]

        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        ### forward diffusion
        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps
        )

        ### compute loss
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = trajectory[condition_mask]
        
        # predict x0 or noise residual
        pred = self.model(noisy_trajectory, timesteps, cond, goal)

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
