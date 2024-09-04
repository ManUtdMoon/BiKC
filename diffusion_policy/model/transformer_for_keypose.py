if __name__ == "__main__":
    import sys
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)

from typing import List, Tuple, Dict
import logging
from einops import rearrange
import torch
import torch.nn as nn
from torch.nn import TransformerDecoderLayer
from diffusion_policy.model.position_embedding import PositionEmbedding2D
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin

logger = logging.getLogger(__name__)

class TransformerForKeypose(ModuleAttrMixin):
    '''Copied from ./diffusion/transformer_for_diffusion.py
    Minor modification:
        - 2d position embedding for image features
        - no time embedding
        - no embedding because all up proj is done in policy encoder
        - no casual-related part because of no multi-step obs
        - input shape is a str-tuple dict, indicating the shape of each input feature
    Reasons for the modification:
        - transformer needs input sequence, but here obs are single frame so we
          have to reshape cnn features before pooling into seq and thus 2d pos emb
        - the dict return is convenient for 2d pos embedding but it is still possible
          to concat rgb and low_dim features respectively in the encoder? (TODO)
    '''
    def __init__(self,
        rgb_keys: List,
        low_dim_keys: List,
        output_dim: int,
        n_layer: int=4,
        n_head: int=4,
        d_embedding: int=256,
        p_drop: float=0.1
    ) -> None:
        super().__init__()

        # input embeddings: all input features should be up projected in encoder
        qpos_emb = nn.Parameter(torch.zeros(1, d_embedding))
        last_keypose_emb = nn.Parameter(torch.zeros(1, d_embedding))
        query_emb = nn.Parameter(torch.zeros(1, d_embedding))
        img_pos_emb = PositionEmbedding2D(d_embedding // 2)

        # main part: a decoder-only transformer
        decoder_layer = TransformerDecoderLayer(
            d_model=d_embedding,
            nhead=n_head,
            dim_feedforward=4*d_embedding,
            dropout=p_drop,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=n_layer
        )

        # output projection
        keypose_ln = nn.LayerNorm(d_embedding)
        keypose_head = nn.Linear(d_embedding, output_dim)

        # save necessary attributes
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.qpos_emb = qpos_emb
        self.last_keypose_emb = last_keypose_emb
        self.query_emb = query_emb
        self.img_pos_emb = img_pos_emb
        self.decoder = decoder
        self.keypose_ln = keypose_ln
        self.keypose_head = keypose_head

        assert len(low_dim_keys) == 2, f"low_dim_keys should be (qpos, last_keypose), but now is {low_dim_keys}"

        # init
        self.apply(self._init_weights)
        logger.info(
            "number of tfm parameters: %.2f M", sum(p.numel() for p in self.parameters()) / 1e6
        )

    def _init_weights(self, module):
        ignore_types = (
            nn.Dropout, 
            PositionEmbedding2D,  
            nn.TransformerDecoderLayer,
            nn.TransformerDecoder,
            nn.ModuleList,
            nn.Sequential
        )
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = [
                'in_proj_weight', 'q_proj_weight', 'k_proj_weight', 'v_proj_weight']
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)
            
            bias_names = ['in_proj_bias', 'bias_k', 'bias_v']
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, TransformerForKeypose):
            torch.nn.init.normal_(module.qpos_emb, mean=0.0, std=0.02)
            torch.nn.init.normal_(module.last_keypose_emb, mean=0.0, std=0.02)
            torch.nn.init.normal_(module.query_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            # no param
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))

    def get_optim_groups(self, weight_decay: float=1e-3):
        '''Copy-pasted from ./diffusion/transformer_for_diffusion.py:
        This function is doing something simple and is being very defensive:
        Separating out all parameters of the model into two buckets: 
        those that will experience weight decay for regularization and 
        those that won't (biases, and layernorm/embedding weights).
        Then return it to the PyTorch optimizer object.
        '''
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (
            torch.nn.Linear,
            torch.nn.MultiheadAttention
        )
        blacklist_weight_modules = (
            torch.nn.LayerNorm,
            torch.nn.Embedding
        )
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.startswith("bias"):
                    # MultiheadAttention bias starts with "bias"
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("qpos_emb")
        no_decay.add("last_keypose_emb")
        no_decay.add("query_emb")
        no_decay.add("_dummy_variable")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups


    def configure_optimizers(self, 
        learning_rate: float=1e-4, 
        weight_decay: float=1e-3,
        betas: Tuple[float, float]=(0.9,0.95)
    ):
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer

    def forward(self,
        obs_feature: Dict
    ) -> torch.Tensor:
        """
        @param obs_feature (Dict):
            str: tensor (B, d1, ..., d_embedding)
        @return next_keypose: (B, output_dim)
        """
        obs_keys = set(obs_feature.keys())
        assert set(self.rgb_keys + self.low_dim_keys) == obs_keys, \
            "keys mismatch: %s vs %s" % (set(self.rgb_keys + self.low_dim_keys), obs_keys)
        
        rgb_keys = self.rgb_keys

        ### get input features and shape, concat them and reshape into a sequence
        bs, h, w, d = obs_feature[rgb_keys[0]].shape
        ncam = len(rgb_keys)
        rgb_pos_emb = self.img_pos_emb(obs_feature[rgb_keys[0]])
        # rgb: pos_emb and concat into (b,ncam,h,w,d)
        rgb_features = torch.stack(
            [obs_feature[k] + rgb_pos_emb for k in rgb_keys],
            dim=1
        )
        assert rgb_features.shape == (bs, ncam, h, w, d), \
            f"rgb shape: {rgb_features.shape}, expected {(bs, ncam, h, w, d)}"
        rgb_features = rearrange(rgb_features, 'b ncam h w d -> b (ncam h w) d')
        # low_dim features (B, D) -> (B, 1, D)
        qpos_features = obs_feature["qpos"] + self.qpos_emb
        last_keypose_features = obs_feature["last_keypose"] + self.last_keypose_emb
        qpos_features = qpos_features.unsqueeze(1)
        last_keypose_features = last_keypose_features.unsqueeze(1)
        
        # obs feature sequence (B, (ncam h w) + 2, D) := (B, L, D)
        context_seq = torch.cat(
            [rgb_features, qpos_features, last_keypose_features],
            dim=1
        )

        ### query (1, D) -> (B, 1, D)
        query_seq = self.query_emb.repeat(bs, 1, 1)

        ### transformer: (B, 1, D), (B, L, D) -> (B, 1, D)
        x = self.decoder(
            tgt=query_seq,
            memory=context_seq
        )
        assert x.shape == (bs, 1, d), f"x shape: {x.shape}, expected {(bs, 1, d)}"
        x = x.squeeze(1) # (B, D)

        ### output projection
        x = self.keypose_ln(x)
        next_keypose = self.keypose_head(x)

        return next_keypose
        

if __name__ == "__main__":
    rgb_keys = ["cam_top", "cam_front"]
    low_dim_keys = ["qpos", "last_keypose"]
    transformer = TransformerForKeypose(
        rgb_keys=rgb_keys,
        low_dim_keys=low_dim_keys,
        output_dim=14
    )
    opt = transformer.configure_optimizers()

    example_input = {
        "cam_top": torch.randn(2, 7, 10, 256),
        "cam_front": torch.randn(2, 7, 10, 256),
        "qpos": torch.randn(2, 256),
        "last_keypose": torch.randn(2, 256)
    }

    next_keypose = transformer(example_input)

    print(next_keypose.shape)  # (2, 14)

    # test with only one camera
    rgb_keys = ["cam_top"]
    low_dim_keys = ["qpos", "last_keypose"]
    transformer = TransformerForKeypose(
        rgb_keys=rgb_keys,
        low_dim_keys=low_dim_keys,
        output_dim=14
    )
    opt = transformer.configure_optimizers()

    example_input = {
        "cam_top": torch.randn(2, 7, 10, 256),
        "qpos": torch.randn(2, 256),
        "last_keypose": torch.randn(2, 256)
    }

    next_keypose = transformer(example_input)

    print(next_keypose.shape)  # (2, 14)