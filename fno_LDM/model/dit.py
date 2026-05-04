# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import Attention, Mlp


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#               Embedding Layers for Reynolds number and Position               #
#################################################################################

class ReynoldsEmbedder(nn.Module):
    """
    Embeds continuous Reynolds numbers into vector representations.
    """
    def __init__(self, max_period=1200, embedding_size=256):
        super().__init__()
        self.frequency_embedding_size = embedding_size
        self.max_period = max_period

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size, max_period=self.max_period)
        return t_freq


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, feature_dim):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, feature_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        feature_dim=64,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        learn_sigma=True,
        layer_num=25,
        cond_dim=1,
        max_cond_period=[1200]
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.num_heads = num_heads
        self.out_channels = 2 if learn_sigma else 1
        self.cond_dim = cond_dim

        self.x_embedder = nn.Linear(feature_dim, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.reynold_embedder = [ReynoldsEmbedder(max_period=max_cond_period[i], embedding_size=hidden_size) for i in range(cond_dim)] if cond_dim > 1 else ReynoldsEmbedder(max_period=max_cond_period[0], embedding_size=hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, layer_num, hidden_size), requires_grad=False)
        
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, feature_dim * self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        
        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.pos_embed.shape[1])
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.weight
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
    
    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs
        return ckpt_forward

    def forward(self, x, t, c):
        """
        Forward pass of DiT.
        x: (N, T, feature_dim) tensor of token sequences
        t: (N,) tensor of diffusion timesteps
        c: (N, cond_dim), tensor of conditional variables
        """
        condition = c
        x = self.x_embedder(x) + self.pos_embed # (N, T, D)
        t = self.t_embedder(t) # (N, D)
        
        if len(condition.shape) == 1:
            assert self.cond_dim == 1, "The number of conditional variables should be equal to 1"
            cond_emb = self.reynold_embedder(condition) # (N, D)
        elif condition.shape[1] == 1:
            assert self.cond_dim == 1, "The number of conditional variables should be equal to 1"
            cond_emb = self.reynold_embedder(condition) # (N, D)
        else:
            assert condition.shape[1] == self.cond_dim, "The number of conditional variables should be equal to the cond_dim"
            cond_emb = torch.sum(torch.stack([self.reynold_embedder[i](condition[:, i]) for i in range(self.cond_dim)], dim=0), dim=0) # (N, D)
        c = t + cond_emb # (N, D)
        
        for block in self.blocks:
            x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, c, use_reentrant=False) # (N, T, D)
        x = self.final_layer(x, c) # (N, T, feature_dim*out_channels)
        
        x = x.view(x.shape[0], x.shape[1], -1, self.out_channels) # (N, T, feature_dim, out_channels)
        x = x.transpose(3, 2) # (N, T, out_channels, feature_dim)
        x = x.contiguous().view(x.shape[0], -1, x.shape[-1]) # (N, T*out_channels, feature_dim)
        
        return x
    
    
#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_1d_sincos_pos_embed(embed_dim, seq_length, cls_token=False, extra_tokens=0):
    """
    seq_length: int of the sequence length
    return:
    pos_embed: [seq_length, embed_dim] or [1+seq_length, embed_dim] (w/ or w/o cls_token)
    """
    position = np.arange(seq_length, dtype=np.float32)

    pos_embed = get_1d_sincos_pos_embed_from_sequence(embed_dim, position)

    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    
    return pos_embed

def get_1d_sincos_pos_embed_from_sequence(embed_dim, position):
    """
    embed_dim: output dimension for each position
    position: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0, "Embedding dimension must be even."

    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    out = np.einsum('m,d->md', position, omega)  # (M, D/2)

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    
    return emb


#################################################################################
#                                   DiT Configs                                 #
#################################################################################

def DiT_XL(**kwargs):
    return DiT(depth=28, hidden_size=1152, num_heads=16, **kwargs)

def DiT_L(**kwargs):
    return DiT(depth=16, hidden_size=1024, num_heads=16, **kwargs)

def DiT_M(**kwargs):
    return DiT(depth=12, hidden_size=768, num_heads=12, **kwargs)

def DiT_S(**kwargs):
    return DiT(depth=12, hidden_size=384, num_heads=6, **kwargs)


DiT_models = {
    'DiT-XL': DiT_XL,
    'DiT-L': DiT_L,
    'DiT-M': DiT_M,
    'DiT-S': DiT_S
}


if __name__ == '__main__':
    # Define the model parameters
    feature_dim = 64
    layer_num = 25
    

    # Create a DiT model instance
    model = DiT_S(
        feature_dim=feature_dim,
        layer_num=layer_num
    )
    
    # Print parameters count
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params / 1e6:.2f}M")

    # Define dummy input data
    batch_size = 2
    seq_length = 25
    x = torch.randn(batch_size, seq_length, feature_dim)  # Random token sequences
    t = torch.randint(0, 1000, (batch_size,))  # Random timesteps
    reynolds = torch.rand(batch_size) * 1000  # Random Reynolds numbers

    # Run a forward pass
    output = model(x, t, reynolds)

    # Print output shape and check it
    print("Output shape:", output.shape)
    expected_shape = (batch_size, seq_length * model.out_channels, feature_dim)
    assert output.shape == expected_shape, f"Expected output shape {expected_shape}, but got {output.shape}"

    print("Test passed!")