import torch
import torch.nn as nn
import math

class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=8, in_chans=3, outer_dim=96, inner_dim=24, inner_stride=4, last_dim=6, last_stride=2):
        super().__init__()

        # Various quantities that help in managing dimensionality across the model 
        self.img_size = img_size
        self.patch_size = patch_size
        num_patches = math.ceil(self.img_size // patch_size) * math.ceil(self.img_size // patch_size)
        self.num_patches = num_patches
        self.inner_dim = inner_dim
        self.last_dim = last_dim
        self.num_words = math.ceil(patch_size / inner_stride) * math.ceil(patch_size / inner_stride)
        self.num_letters = math.ceil(inner_stride / last_stride) * math.ceil(inner_stride / last_stride)
        inner_size = inner_stride
        self.inner_size = inner_size

        self.unfold = nn.Unfold(kernel_size=patch_size, stride=self.patch_size)
        #self.proj_inner = nn.Conv2d(in_chans, last_dim, kernel_size=3, padding=1, stride=last_stride)
        self.proj_inner = nn.Conv2d(in_chans, last_dim, kernel_size=1, padding=0, stride=last_stride)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})."
        #Do patch splits till we get letter embeddings, and apply conv2d to extract image features
        x = self.unfold(x) # B, Ck2, N
        x = x.transpose(1, 2)
        x = x.reshape(B * self.num_patches * self.num_words, C, self.inner_size, self.inner_size) # B*N*M, C, r, r
        x = self.proj_inner(x) # B*N*M, ci, r, r
        x = x.reshape(B * self.num_patches * self.num_words, self.last_dim, -1)
        x = x.transpose(1, 2) # B*N*M, r*r, ci
        return x
