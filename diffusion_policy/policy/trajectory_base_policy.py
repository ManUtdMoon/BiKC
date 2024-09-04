from typing import Dict, Union
import torch
import torch.nn as nn
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.model.common.normalizer import LinearNormalizer

class TrajectoryBasePolicy(ModuleAttrMixin):
    # init accepts keyword argument shape_meta, see config/task/*_image.yaml
    # may define obs_encoder

    def predict_trajectory(
        self,
        obs_dict: Dict[str, torch.Tensor],
        next_keypose: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        @params
            obs_dict:
                str: B,To,* 
            next_keypose: B,Da
        @return:
            dict:
                trajectory: B,Ta,Da
                trajectory_pred: B,H,Da
        """
        raise NotImplementedError()

    # reset state for stateful policies
    def reset(self):
        pass

    # ========== training ===========
    # no standard training interface except setting normalizer
    def set_normalizer(self, normalizer: LinearNormalizer):
        raise NotImplementedError()
    
    def compute_loss(
        self,
        batch: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]],
    ) -> torch.Tensor:
        """
        batch: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]
            obs_dict:
                str: B,To,*
            next_keypose: B,Da
            action: B,Ta,Da
        return: torch.Tensor
        """
        raise NotImplementedError()
