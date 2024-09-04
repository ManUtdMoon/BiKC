if __name__ == "__main__":
    import sys
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent.parent)
    sys.path.append(ROOT_DIR)

from typing import Union, Optional, Tuple
import logging
import torch
import torch.nn as nn
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin

logger = logging.getLogger(__name__)

class GoalCondTransformerForDiffusion(ModuleAttrMixin):
    def __init__(self,
        # task-related
        input_dim: int,
        output_dim: int,
        horizon: int,
        n_obs_steps: int = None,
        obs_dim: int = 0,
        # tfm arch
        n_layer: int = 4,
        n_head: int = 4,
        d_embedding: int=256,
        p_drop: float = 0.1,
        # mask
        causal_attn: bool=True,
        obs_as_cond: bool=True,
    ) -> None:
        super().__init__()

        # compute number of tokens for main trunk and condition encoder
        if n_obs_steps is None:
            n_obs_steps = horizon
        
        T = horizon
        T_cond = 1 + 1 # diffusion step + next_keypose
        assert obs_as_cond and (obs_dim > 0)
        T_cond += n_obs_steps

        # input embedding
        self.sample_emb = nn.Linear(input_dim, d_embedding)
        self.sample_pos_emb = nn.Parameter(torch.zeros(1, T, d_embedding))

        # cond encoder
        self.time_emb = SinusoidalPosEmb(d_embedding)
        self.goal_emb = nn.Linear(input_dim, d_embedding)
        self.obs_emb = nn.Linear(obs_dim, d_embedding)

        self.cond_pos_emb = nn.Parameter(torch.zeros(1, T_cond, d_embedding))

        ### transformer part
        self.encoder = None
        self.decoder = None
        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(d_embedding, 4 * d_embedding),
            nn.Mish(),
            nn.Linear(4 * d_embedding, d_embedding)
        )
        # decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_embedding,
            nhead=n_head,
            dim_feedforward=4*d_embedding,
            dropout=p_drop,
            activation='gelu',
            batch_first=True,
            norm_first=True # important for stability
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=n_layer
        )

        # attention mask
        if causal_attn:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            # torch.nn.Transformer uses additive mask as opposed to multiplicative mask in minGPT
            # therefore, the upper triangle should be -inf and others (including diag) should be 0.
            sz = T
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            self.register_buffer("mask", mask)
            
            if obs_as_cond:
                S = T_cond
                t, s = torch.meshgrid(
                    torch.arange(T),
                    torch.arange(S),
                    indexing='ij'
                )
                mask = t >= (s-2) # add two dims: 1st is time and 2nd is goal
                mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
                self.register_buffer('memory_mask', mask)
            else:
                self.memory_mask = None
        else:
            self.mask = None
            self.memory_mask = None

        # decoder head
        self.ln_f = nn.LayerNorm(d_embedding)
        self.head = nn.Linear(d_embedding, output_dim)
            
        # constants
        self.T = T
        self.T_cond = T_cond
        self.horizon = horizon
        self.obs_as_cond = obs_as_cond

        # init
        self.apply(self._init_weights)
        logger.info(
            "number of parameters: %.2f M", sum(p.numel() for p in self.parameters()) / 1e6
        )

    def _init_weights(self, module):
        ignore_types = (
            nn.Dropout, 
            SinusoidalPosEmb, 
            nn.TransformerDecoderLayer,
            nn.TransformerDecoder,
            nn.ModuleList,
            nn.Mish,
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
        elif isinstance(module, GoalCondTransformerForDiffusion):
            torch.nn.init.normal_(module.sample_pos_emb, mean=0.0, std=0.02)
            torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            # no param
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))
    
    def get_optim_groups(self, weight_decay: float=1e-3):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
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
        no_decay.add("sample_pos_emb")
        no_decay.add("cond_pos_emb")
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
            betas: Tuple[float, float]=(0.9,0.95)):
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer

    def forward(self, 
        sample: torch.Tensor, 
        timestep: Union[torch.Tensor, float, int], 
        cond: Optional[torch.Tensor]=None,
        goal: torch.Tensor=None,
        **kwargs
    ):
        """
        @params
            x: B,T,Da
            t: B, or int (avoid if possible)
            cond (obs): B,To,Do
            goal (next_keypose): B,Da

            **note: 
            - T_cond = 1 (t) + 1 (goal) + To
            - D := d_embedding
        @return
            output: B,T,Da - sample, not noise
        """
        ### input embedding
        # diffusion timestep
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
        time_feats = self.time_emb(timesteps).unsqueeze(1)
        # (B,1,D)

        # goal embedding
        goal_feats = self.goal_emb(goal).unsqueeze(1) # (B,1,D)

        # obs embedding
        obs_feats = self.obs_emb(cond)  # (B,To,D)
        
        ### condition fusion and position embedding
        context_feats = torch.cat(
            [time_feats, goal_feats, obs_feats], dim=1
        ) # (B,T_cond,D)
        tc = context_feats.shape[1]
        context_pe = self.cond_pos_emb[:, :tc, :]
        context_seq = self.encoder(context_feats + context_pe) # (B,Tc,D)
        
        ### decoder
        # sample embedding
        sample_feats = self.sample_emb(sample)
        t = sample_feats.shape[1]
        sample_pe = self.sample_pos_emb[:, :t, :]
        sample_seq = sample_feats + sample_pe # (B,T,D)
        x = self.decoder(
            tgt=sample_seq,
            memory=context_seq,
            tgt_mask=self.mask,
            memory_mask=self.memory_mask
        )
        # (B,T,D)
        
        ### head
        x = self.ln_f(x)
        x = self.head(x) # (B,T,D_out)
        return x

if __name__ == "__main__":
    tfm = GoalCondTransformerForDiffusion(
        input_dim=14,
        output_dim=14,
        horizon=16,
        n_obs_steps=2,
        obs_dim=20,
    )

    opt = tfm.configure_optimizers()

    t = torch.tensor(2)
    sample = torch.randn(2, 16, 14)
    cond = torch.randn(2, 2, 20)
    next_keypose = torch.randn(2, 14)

    out = tfm(sample, t, cond, next_keypose)
    print(out.shape)
