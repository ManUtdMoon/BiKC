if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent.parent)
    sys.path.append(ROOT_DIR)

from typing import Dict, Tuple, Union
import logging
import copy
import torch
import torch.nn as nn
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
from einops import rearrange
from diffusion_policy.model.vision.crop_randomizer import CropRandomizer
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules

logger = logging.getLogger(__name__)

class DictImageObsEncoder(ModuleAttrMixin):
    def __init__(self,
            shape_meta: dict,
            rgb_model: Union[nn.Module, Dict[str,nn.Module]],
            resize_shape: Union[Tuple[int,int], Dict[str,tuple], None]=None,
            crop_shape: Union[Tuple[int,int], Dict[str,tuple], None]=None,
            random_crop: bool=True,
            # replace BatchNorm with GroupNorm
            use_group_norm: bool=False,
            # use single rgb model for all rgb inputs
            share_rgb_model: bool=False,
            # renormalize rgb input with imagenet normalization
            # assuming input in [0,1]
            imagenet_norm: bool=False,
            # out layer of resnet
            cnn_out_layer: str="layer4",
            # proj for transformer
            embedding_dim: int=256
        ):
        """
        Assumes rgb input: B,C,H,W
        Assumes low_dim input: B,D
        """
        super().__init__()

        rgb_keys = list()
        low_dim_keys = list()
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_shape_map = dict()
        rgbkey_downsize_model_map = nn.ModuleDict()

        # handle sharing vision backbone
        if share_rgb_model:
            assert isinstance(rgb_model, nn.Module)
            key_model_map["rgb"] = IntermediateLayerGetter(
                rgb_model,
                return_layers={cnn_out_layer: "0"}
            )
            rgbkey_downsize_model_map["rgb"] = nn.Conv2d(
                rgb_model.num_channels, embedding_dim, kernel_size=1
            )

        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            key_shape_map[key] = shape
            if type == 'rgb':
                rgb_keys.append(key)
                # configure model for this key
                this_model = None
                if not share_rgb_model:
                    if isinstance(rgb_model, dict):
                        # have provided model for each key
                        this_model = rgb_model[key]
                    else:
                        assert isinstance(rgb_model, nn.Module)
                        # have a copy of the rgb model
                        this_model = copy.deepcopy(rgb_model)
                
                if this_model is not None:
                    if use_group_norm:
                        this_model = replace_submodules(
                            root_module=this_model,
                            predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                            func=lambda x: nn.GroupNorm(
                                num_groups=x.num_features//16, 
                                num_channels=x.num_features)
                        )
                    num_channels = _get_last_layer_channels(this_model)
                    # make the output the last conv layer before avg pooling
                    this_model = IntermediateLayerGetter(
                        this_model,
                        return_layers={cnn_out_layer: "0"}
                    )
                    key_model_map[key] = this_model
                    try:
                        rgbkey_downsize_model_map[key] = nn.Conv2d(
                            num_channels, embedding_dim, kernel_size=1
                        )
                    except:
                        import pdb; pdb.set_trace()
                
                # configure resize
                input_shape = shape
                this_resizer = nn.Identity()
                if resize_shape is not None:
                    if isinstance(resize_shape, dict):
                        h, w = resize_shape[key]
                    else:
                        h, w = resize_shape
                    this_resizer = torchvision.transforms.Resize(
                        size=(h,w)
                    )
                    input_shape = (shape[0],h,w)

                # configure randomizer
                this_randomizer = nn.Identity()
                if crop_shape is not None:
                    if isinstance(crop_shape, dict):
                        h, w = crop_shape[key]
                    else:
                        h, w = crop_shape
                    if random_crop:
                        this_randomizer = CropRandomizer(
                            input_shape=input_shape,
                            crop_height=h,
                            crop_width=w,
                            num_crops=1,
                            pos_enc=False
                        )
                    else:
                        this_randomizer = torchvision.transforms.CenterCrop(
                            size=(h,w)
                        )
                # configure normalizer
                this_normalizer = nn.Identity()
                if imagenet_norm:
                    this_normalizer = torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                
                this_transform = nn.Sequential(this_resizer, this_randomizer, this_normalizer)
                key_transform_map[key] = this_transform
            elif type == 'low_dim':
                low_dim_keys.append(key)
                # up projection for low dim input
                key_model_map[key] = nn.Linear(shape[0], embedding_dim)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")
        rgb_keys = sorted(rgb_keys)
        low_dim_keys = sorted(low_dim_keys)

        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.share_rgb_model = share_rgb_model
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.key_shape_map = key_shape_map
        self.rgbkey_downsize_model_map = rgbkey_downsize_model_map

        logger.info(
            "number of encoder parameters: %.2f M", sum(p.numel() for p in self.parameters()) / 1e6
        )

    def forward(self, obs_dict):
        '''
        @param obs_dict: dict of tensors
            str: tensor (B, D1, ...)
        @return: result_dict: dict of tensors
            str: tensor (B, d1, ..., embedding_dim)
        '''
        batch_size = None
        result = dict()
        # process rgb input
        if self.share_rgb_model:
            # pass all rgb obs to rgb model
            imgs = list()
            for key in self.rgb_keys:
                img = obs_dict[key]
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                assert img.shape[1:] == self.key_shape_map[key]
                img = self.key_transform_map[key](img)
                imgs.append(img)
            # (N*B,C,H,W)
            imgs = torch.cat(imgs, dim=0)
            # (N*B,D,h1,w1): because we take feature out of last conv layer
            feature = self.key_model_map["rgb"](imgs)["0"]
            # (N*B,d,h1,w1): downsize to embedding_dim
            feature = self.rgbkey_downsize_model_map["rgb"](feature)
            # (N,B,h1,w1,d)
            feature = rearrange(feature, '(n b) c h w -> n b h w c', b=batch_size)
            for i, key in enumerate(self.rgb_keys):
                result[key] = feature[i]
        else:
            # run each rgb obs to independent models
            for key in self.rgb_keys:
                img = obs_dict[key]
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                assert img.shape[1:] == self.key_shape_map[key], f"{img.shape[1:]} != {self.key_shape_map[key]}"
                img = self.key_transform_map[key](img)
                # get feature from last conv layer
                feature = self.key_model_map[key](img)["0"]
                # downsize to embedding_dim
                feature = self.rgbkey_downsize_model_map[key](feature)
                feature = rearrange(feature, 'b c h w -> b h w c')
                result[key] = feature
        
        # process lowdim input
        for key in self.low_dim_keys:
            data = obs_dict[key]
            if batch_size is None:
                batch_size = data.shape[0]
            else:
                assert batch_size == data.shape[0]
            assert data.shape[1:] == self.key_shape_map[key]
            result[key] = self.key_model_map[key](data)

        return result
    
    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict()
        output_shape = dict()
        obs_shape_meta = self.shape_meta['obs']
        batch_size = 1
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            this_obs = torch.zeros(
                (batch_size,) + shape, 
                dtype=self.dtype,
                device=self.device)
            example_obs_dict[key] = this_obs
        example_output = self.forward(example_obs_dict)
        for key, attr in obs_shape_meta.items():
            output_shape[key] = example_output[key].shape[1:]
        return output_shape

def _get_last_layer_channels(resnet_model):
    # Access the last block of layer4
    last_block = resnet_model.layer4[-1]
    
    # ResNet50 and above use Bottleneck blocks, whereas ResNet18 and ResNet34 use BasicBlock.
    # Both block types have a 'conv3' in Bottleneck and a 'conv2' in BasicBlock as their last convolutional layer.
    if hasattr(last_block, 'conv3'):
        last_conv_layer = last_block.conv3
    elif hasattr(last_block, 'conv2'):
        last_conv_layer = last_block.conv2
    else:
        raise TypeError("Unsupported ResNet block type.")
    
    # Get the number of output channels
    num_channels = last_conv_layer.out_channels
    
    return num_channels

if __name__ == "__main__":
    shape_meta = {
        "obs": {
            "cam_top": {"shape": (3, 480, 640), "type": "rgb"},
            "cam_front": {"shape": (3, 480, 640), "type": "rgb"},
            "qpos": {"shape": (14,), "type": "low_dim"},
            "last_keypose": {"shape": (14,), "type": "low_dim"},
        }
    }
    rgb_model = torchvision.models.resnet18()
    rgb_model.fc = nn.Identity()
    rgb_model.num_channels = 512
    resize_shape = [240, 320]
    crop_shape = [220, 300]

    example_obs_dict = dict()
    obs_shape_meta = shape_meta['obs']
    batch_size = 1
    for key, attr in obs_shape_meta.items():
        shape = tuple(attr['shape'])
        this_obs = torch.randn(
            (batch_size,) + shape, 
            dtype=torch.float32,
            device='cpu')
        example_obs_dict[key] = this_obs

    encoder = DictImageObsEncoder(
        shape_meta=shape_meta,
        rgb_model=rgb_model,
        resize_shape=resize_shape,
        crop_shape=crop_shape,
        random_crop=False,
        use_group_norm=True,
        share_rgb_model=True,
        imagenet_norm=True,
        cnn_out_layer="layer4",
        embedding_dim=256
    )

    example_output = encoder(example_obs_dict)
    for key, value in example_output.items():
        print(key, value.shape)

    output_shape = encoder.output_shape()
    print(output_shape)

    example_obs_dict["cam_top"] = torch.randn(
        (batch_size, 3, 480, 640), 
        dtype=torch.float32,
    )
    example_obs_dict["cam_front"] = example_obs_dict["cam_top"].clone()
    assert torch.allclose(
        example_obs_dict["cam_top"], 
        example_obs_dict["cam_front"],
        atol=1e-3
    ), print(example_obs_dict["cam_top"][0,0,0,:10], "\n", example_obs_dict["cam_front"][0,0,0,:10])

    example_output = encoder(example_obs_dict)
    assert torch.allclose(
        example_output["cam_top"], 
        example_output["cam_front"],
        atol=1e-3
    ), print(example_output["cam_top"][0,0,0,:10], "\n", example_output["cam_front"][0,0,0,:10])

