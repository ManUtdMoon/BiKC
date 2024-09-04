from typing import Dict, Union
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.keypose_base_policy import KeyposeBasePolicy
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply

logger = logging.getLogger(__name__)

class KeyposeCNNMLPPolicy(KeyposeBasePolicy):
    def __init__(self,
            shape_meta: dict,
            obs_encoder: MultiImageObsEncoder,
            # mlp architecture
            hidden_depth: int=2,
            hidden_dim: int=1024):
        super().__init__()

        # parse shapes
        next_keypose_shape = shape_meta['next_keypose']['shape']
        assert len(next_keypose_shape) == 1
        next_keypose_dim = next_keypose_shape[0]
        # get feature dim
        obs_feature_dim = obs_encoder.output_shape()[0]

        # create mlp
        model = mlp(
            input_dim=obs_feature_dim,
            hidden_dim=hidden_dim,
            output_dim=next_keypose_dim,
            hidden_depth=hidden_depth
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.normalizer = LinearNormalizer()
        self.obs_feature_dim = obs_feature_dim

        logger.info(
            "number of policy parameters: %.2f M", sum(p.numel() for p in model.parameters()) / 1e6
        )

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


def mlp(input_dim, hidden_dim, output_dim, hidden_depth):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    trunk = nn.Sequential(*mods)
    return trunk