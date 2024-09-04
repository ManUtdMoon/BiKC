from typing import Dict, Union
import torch
import torch.nn as nn
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.model.common.normalizer import LinearNormalizer

class KeyposeBasePolicy(ModuleAttrMixin):
    # init accepts keyword argument shape_meta, see config/task/*_image.yaml
    # may define obs_encoder

    def predict_next_keypose(
        self,
        obs_dict: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        obs_dict:
            str: B,*
        return: B,Da
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
                str: B,*
            next_keypose: B,Da
        return: torch.Tensor
        """
        raise NotImplementedError()
