import torch
import torch.nn as nn

from .patch_embed import PatchEmbed
from .blocks import Block
from .utils import create_projection_layers


class RTNT(nn.Module):
    def __init__(self, img_size=32, patch_size=8, in_chans=1, num_classes=10, outer_dim=96, inner_dim=24,
                 depth=12, outer_num_heads=3, inner_num_heads=2, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., norm_layer=nn.LayerNorm, inner_stride=4, se=0,
                 last_dim=6, last_stride=2, last_num_heads=1):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.outer_dim = outer_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, outer_dim=outer_dim,
            inner_dim=inner_dim, inner_stride=inner_stride, last_dim=last_dim, last_stride=last_stride)

        # Qunatities that control dimensionality across the model
        self.num_patches = num_patches = self.patch_embed.num_patches
        self.num_words = num_words = self.patch_embed.num_words
        self.num_letters = num_letters = self.patch_embed.num_letters

        # Classfication tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, outer_dim))

        #i Intializations for outer abd middle tokens, all set to zeros
        self.outer_tokens = nn.Parameter(torch.zeros(1, num_patches, outer_dim), requires_grad=False)
        self.inner_tokens = nn.Parameter(torch.zeros(num_patches, num_words, inner_dim), requires_grad=False)

        # Positional embeddings
        self.outer_pos = nn.Parameter(torch.zeros(1, num_patches + 1, outer_dim))
        self.inner_pos = nn.Parameter(torch.zeros(1, num_words, inner_dim))
        self.last_pos = nn.Parameter(torch.zeros(1, num_letters, last_dim))

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Layers to help projecting letter/word embedding on word/sentence embeddings
        self.middle_tokens_proj_norm1, self.middle_tokens_proj, self.middle_tokens_proj_norm2 = create_projection_layers(num_words, inner_dim, outer_dim, norm_layer)
        self.inner_tokens_proj_norm1, self.inner_tokens_proj, self.inner_tokens_proj_norm2 = create_projection_layers(num_letters, last_dim, inner_dim, norm_layer)


        #Yoy can control which layers include the inner transformer so to mimic TNT using this model
        vanilla_idxs = []
        blocks = []
        for i in range(depth):



            if i in vanilla_idxs:
                blocks.append(Block(
                    outer_dim=outer_dim, outer_num_heads=outer_num_heads, middle_dim=-1, middle_num_heads=inner_num_heads,
                    num_words=num_words, inner_dim=-1, innet_num_heads=last_num_heads, num_letters=num_letters,
                    mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                    attn_drop=attn_drop_rate, norm_layer=norm_layer, se=se))
            else:
                blocks.append(Block(
                    outer_dim=outer_dim, outer_num_heads=outer_num_heads, middle_dim=inner_dim, middle_num_heads=inner_num_heads,
                    num_words=num_words, inner_dim=last_dim, inner_num_heads=last_num_heads, num_letters=num_letters,
                    mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                    attn_drop=attn_drop_rate, norm_layer=norm_layer, se=se))
                
        self.blocks = nn.ModuleList(blocks)
        self.norm = norm_layer(outer_dim)

        # Classifier head
        self.head = nn.Linear(outer_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        last_tokens = self.patch_embed(x) + self.last_pos

        # Set the inner and outer tokens with image data instead of zeros/normal distripution
        inner_tokens = self.inner_tokens_proj_norm2(self.inner_tokens_proj(self.inner_tokens_proj_norm1(last_tokens.reshape(B*self.num_patches, self.num_words, -1)))) + self.inner_pos # B*N, s*s, c
        outer_tokens = self.middle_tokens_proj_norm2(self.middle_tokens_proj(self.middle_tokens_proj_norm1(inner_tokens.reshape(B, self.num_patches, -1)))) # B, p*p, C

        outer_tokens = torch.cat((self.cls_token.expand(B, -1, -1), outer_tokens), dim=1)
        outer_tokens = outer_tokens + self.outer_pos
        outer_tokens = self.pos_drop(outer_tokens)

        for blk in self.blocks:
            last_tokens, inner_tokens, outer_tokens = blk(last_tokens, inner_tokens, outer_tokens)

        outer_tokens = self.norm(outer_tokens)
        return outer_tokens[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
    
    # Class Methods
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.outer_dim, num_classes) if num_classes > 0 else nn.Identity()