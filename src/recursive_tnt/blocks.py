import torch
import torch.nn as nn


from .attention import Attention
from .mlp import Mlp
from .se import SE
from .utils import create_projection_layers


# Droppaths in the original implementation are not implemented

class Block(nn.Module):

    def __init__(self, outer_dim, outer_num_heads, middle_dim, middle_num_heads, num_words, inner_dim, inner_num_heads, num_letters, #Positional arguments
                 mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., act_layer=nn.GELU, #keyword arguments
                 norm_layer=nn.LayerNorm, se=0):
        super().__init__()

        # Check if that block should have inner transformer or ignore it
        self.has_inner = inner_dim > 0
        if self.has_inner:
            # Inner
            self.last_norm1 = norm_layer(inner_dim)
            self.last_attn = Attention(
                inner_dim, inner_dim, num_heads=inner_num_heads, qkv_bias=qkv_bias,
                qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            
            self.last_norm2 = norm_layer(inner_dim)
            self.last_mlp = Mlp(in_features=inner_dim, hidden_features=int(inner_dim * mlp_ratio),
                                 out_features=inner_dim, act_layer=act_layer, drop=drop)

            
            # Create layers that will be used to project the inner embedding on the middle embedding
            self.inner_tokens_proj_norm1, self.inner_tokens_proj, self.inner_tokens_proj_norm2 = create_projection_layers(num_letters, inner_dim, middle_dim, norm_layer)

        # Check if that block should have a middle transformer or ignore it
        self.has_middle = middle_dim > 0
        if self.has_middle:
            # Middle
            self.middle_norm1 = norm_layer(middle_dim)
            self.middle_attn = Attention(
                middle_dim, middle_dim, num_heads=middle_num_heads, qkv_bias=qkv_bias,
                qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            
            self.middle_norm2 = norm_layer(middle_dim)
            self.middle_mlp = Mlp(in_features=middle_dim, hidden_features=int(middle_dim * mlp_ratio),
                                 out_features=middle_dim, act_layer=act_layer, drop=drop)

            # Create layers that will be used to project the middle embedding to the outer embedding
            self.middle_tokens_proj_norm1, self.middle_tokens_proj, self.middle_tokens_proj_norm2 = create_projection_layers(num_words, middle_dim, outer_dim, norm_layer)


        # Outer
        self.outer_norm1 = norm_layer(outer_dim)
        self.outer_attn = Attention(
            outer_dim, outer_dim, num_heads=outer_num_heads, qkv_bias=qkv_bias,
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
        self.outer_norm2 = norm_layer(outer_dim)
        self.outer_mlp = Mlp(in_features=outer_dim, hidden_features=int(outer_dim * mlp_ratio),
                             out_features=outer_dim, act_layer=act_layer, drop=drop)
        

        # SE
        self.se = se
        self.se_layer = None
        if self.se > 0:
            self.se_layer = SE(outer_dim, 0.25 ) #Ratio of TNT


    def forward(self, inner_tokens, middle_tokens, outer_tokens):
        if self.has_inner:
            inner_tokens = inner_tokens + self.last_attn(self.last_norm1(inner_tokens)) # B*N*M, s*s, ci
            inner_tokens = inner_tokens + self.last_mlp(self.last_norm2(inner_tokens)) # B*N*M, s*s, ci
            BN, M, cm = middle_tokens.size()
            # Concatenate the inner embeddings then project on middle embeddings
            middle_tokens = middle_tokens + self.inner_tokens_proj_norm2(self.inner_tokens_proj(self.inner_tokens_proj_norm1(inner_tokens.reshape(BN, M, -1)))) # B*N, M, cm
        if self.has_middle:
            middle_tokens = middle_tokens + self.middle_attn(self.middle_norm1(middle_tokens)) # B*N, k*k, c
            middle_tokens = middle_tokens + self.middle_mlp(self.middle_norm2(middle_tokens)) # B*N, k*k, c
            B, N, C = outer_tokens.size()
            # Concatenate the middle embeddings then project on outer embeddings (NOTE that the array slice is due to the class token)
            outer_tokens[:,1:] = outer_tokens[:,1:] + self.middle_tokens_proj_norm2(self.middle_tokens_proj(self.middle_tokens_proj_norm1(middle_tokens.reshape(B, N-1, -1)))) # B, N, C
        if self.se > 0:
            outer_tokens = outer_tokens + self.outer_attn(self.outer_norm1(outer_tokens))
            tmp_outer_tokens = self.outer_mlp(self.outer_norm2(outer_tokens))
            outer_tokens = outer_tokens + tmp_outer_tokens + self.se_layer(tmp_outer_tokens)
        else:
            outer_tokens = outer_tokens + self.outer_attn(self.outer_norm1(outer_tokens))
            outer_tokens = outer_tokens + self.outer_mlp(self.outer_norm2(outer_tokens))
        return inner_tokens, middle_tokens, outer_tokens
