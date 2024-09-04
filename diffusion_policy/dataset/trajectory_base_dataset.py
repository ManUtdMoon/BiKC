from typing import Dict

import torch
import torch.nn
from diffusion_policy.model.common.normalizer import LinearNormalizer


class TrajectoryBaseDataset(torch.utils.data.Dataset):
    def get_validation_dataset(self):
        # return an empty dataset by default
        return TrajectoryBaseDataset()

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        raise NotImplementedError()
    
    def __len__(self) -> int:
        return 0
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        output:
            obs: 
                key: To, Do
            next_keypose: Dk
            action: Horizon, Da
        """
        raise NotImplementedError()
