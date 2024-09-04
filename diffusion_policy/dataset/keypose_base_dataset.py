from typing import Dict

import torch
import torch.nn
from diffusion_policy.model.common.normalizer import LinearNormalizer


class KeyposeBaseDataset(torch.utils.data.Dataset):
    def get_validation_dataset(self):
        # return an empty dataset by default
        return KeyposeBaseDataset()

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        raise NotImplementedError()
    
    def __len__(self) -> int:
        return 0
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        output:
            obs: 
                key: T, *
            next_keypose: T, Da
        """
        raise NotImplementedError()
