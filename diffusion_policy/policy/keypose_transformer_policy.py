from typing import Dict, Union, Tuple
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.keypose_base_policy import KeyposeBasePolicy
from diffusion_policy.model.vision.dict_image_obs_encoder import DictImageObsEncoder
from diffusion_policy.model.transformer_for_keypose import TransformerForKeypose
from diffusion_policy.common.pytorch_util import dict_apply

logger = logging.getLogger(__name__)

class KeyposeTransformerPolicy(KeyposeBasePolicy):
    def __init__(self,
        shape_meta: dict,
        obs_encoder: DictImageObsEncoder,
        # transformer architecture
        n_layer: int=4,
        n_head: int=4,
        d_embedding: int=256,
        p_drop: float=0.1,
    ) -> None:
        super().__init__()

        # parse shapes
        next_keypose_shape = shape_meta['next_keypose']['shape']
        assert len(next_keypose_shape) == 1
        next_keypose_dim = next_keypose_shape[0]
        # get feature dim dict
        obs_feature_dim_dict = obs_encoder.output_shape()
        rgb_keys = obs_encoder.rgb_keys
        low_dim_keys = obs_encoder.low_dim_keys

        # create transformer model
        model = TransformerForKeypose(
            rgb_keys=rgb_keys,
            low_dim_keys=low_dim_keys,
            output_dim=next_keypose_dim,
            n_layer=n_layer,
            n_head=n_head,
            d_embedding=d_embedding,
            p_drop=p_drop
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.normalizer = LinearNormalizer()
        self.obs_feature_dim_dict = obs_feature_dim_dict

    # ========== inference ==========
    def predict_next_keypose(
        self,
        obs_dict: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        @param obs_dict
            keys = shape_meta["obs"].keys()
        @return next_keypose: (B, next_keypose_dim)
        """
        # normalize obs
        nobs = self.normalizer.normalize(obs_dict)
        # encode obs
        nobs_feature = self.obs_encoder(nobs)

        # predict normalized next_keypose
        nnext_keypose = self.model(nobs_feature)

        # unnormalize next_keypose
        next_keypose = self.normalizer["next_keypose"].unnormalize(nnext_keypose)

        return next_keypose

    # ========== training ==========
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

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(
        self,
        batch: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]],
    ) -> torch.Tensor:
        nobs = self.normalizer.normalize(batch["obs"])
        target = self.normalizer["next_keypose"].normalize(batch["next_keypose"]).detach()

        # encode obs
        nobs_feature = self.obs_encoder(nobs)

        # predict normalized next_keypose
        pred = self.model(nobs_feature)

        # compute loss
        loss = F.mse_loss(pred, target)
        return loss
