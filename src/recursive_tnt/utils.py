import torch
import torch.nn as nn

def create_projection_layers(num_of_words, inner_dim, outer_dim, norm_layer):
    """
    Creates three layers for projection:
    1. First normalization layer
    2. Linear projection layer
    3. Second normalization layer
    
    Args:
        num_of_words: Number of patches in the inner that consists the outer
        inner_dim: Inner dimension size
        outer_dim: Outer dimension size
        norm_layer: Normalization layer function/class to use
        
    Returns:
        Tuple of (proj_norm1_last, proj_last, proj_norm2_last)
    """
    normalization_layer = norm_layer(num_of_words * inner_dim)
    projection_layer = nn.Linear(num_of_words * inner_dim, outer_dim, bias=False)
    normalization_layer_after_projection = norm_layer(outer_dim)
    
    return normalization_layer, projection_layer, normalization_layer_after_projection