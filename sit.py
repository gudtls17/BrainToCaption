# -*- coding: utf-8 -*-
# @Author: Your name
# @Date:   1970-01-01 01:00:00
# @Last Modified by:   Your name
# @Last Modified time: 2024-01-15 10:00:00
#
# Modified for tokenized input (batch, token, dim)
#
# Original SiT implementation by Simon Dahan @SD3004
# Copyright (c) 2021 MeTrICS Lab
#

'''
This file contains our implementation of the ViT model adapted for tokenized fMRI input.
Original: https://arxiv.org/abs/2010.11929

Key changes:
- Input shape: (batch, num_tokens, token_dim) instead of (batch, channels, patches, vertices)
- Simplified embedding layer for pre-tokenized input
'''

import torch
from torch import nn
import numpy as np

from einops import repeat

from vit_pytorch.vit import Transformer

from timm.models.layers import trunc_normal_


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    Generate 1D sinusoidal positional embeddings
    
    Args:
        embed_dim: output dimension for each position
        pos: positions to be encoded: [num_positions]
    
    Returns:
        pos_embed: [num_positions, embed_dim]
    """
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class SiT(nn.Module):
    """
    Surface Transformer for tokenized fMRI input
    
    Input shape: (batch, num_tokens, token_dim)
    Output shape: (batch, num_classes)
    """
    
    def __init__(
        self, 
        *,
        dim,                    # Hidden dimension of transformer
        depth,                  # Number of transformer layers
        heads,                  # Number of attention heads
        pool='cls',             # Pooling type: 'cls' or 'mean'
        num_tokens=1000,        # Number of input tokens
        num_classes=1,          # Number of output classes
        token_dim=768,          # Dimension of input tokens
        dim_head=64,            # Dimension per attention head
        dropout=0.,             # Dropout rate
        emb_dropout=0.,         # Embedding dropout rate
        bottleneck_dropout=0.,  # Bottleneck dropout rate
        mlp_ratio=4,            # MLP expansion ratio
        use_pe='sin-cos',       # Positional embedding type: 'sin-cos', 'trainable', or False
        use_confounds=False,    # Whether to use confounds
        use_bottleneck=False,   # Whether to use bottleneck projection
        weights_layers_init=False,  # Use timm-style weight initialization
        use_class_token=True,   # Whether to use CLS token
        trainable_pos_emb=True, # Whether positional embeddings are trainable
        no_class_token_emb=True,  # Whether to exclude CLS token from pos embeddings
    ):
        super().__init__()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        
        # Store configuration
        self.use_pe = use_pe
        self.use_confounds = use_confounds
        self.num_tokens = num_tokens
        self.encoding_dim = dim
        self.use_class_token = use_class_token
        self.no_class_token_emb = no_class_token_emb
        self.num_prefix_tokens = 1 if use_class_token else 0
        self.pool = pool

        # ====================================================================
        # TOKEN EMBEDDING LAYER
        # ====================================================================
        # Project input tokens from token_dim to hidden dim
        
        if use_bottleneck:
            print(f'Using bottleneck: {token_dim} -> 1024 -> {dim}')
            self.to_patch_embedding = nn.Sequential(
                nn.Dropout(bottleneck_dropout),
                nn.Linear(token_dim, 1024),
                nn.GELU(),
                nn.Dropout(bottleneck_dropout),
                nn.Linear(1024, dim),
            )
        else:
            # Simple linear projection
            self.to_patch_embedding = nn.Linear(token_dim, dim)

        # ====================================================================
        # OPTIONAL CONFOUNDS PROJECTION
        # ====================================================================
        if use_confounds:
            self.proj_confound = nn.Sequential(
                nn.BatchNorm1d(1),
                nn.Linear(1, dim)
            )
        
        self.dropout = nn.Dropout(emb_dropout)

        # ====================================================================
        # TRANSFORMER
        # ====================================================================
        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_ratio * dim,
            dropout=dropout
        )

        # ====================================================================
        # OUTPUT HEAD
        # ====================================================================
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        
        # ====================================================================
        # POSITIONAL EMBEDDINGS
        # ====================================================================
        if self.use_pe == 'trainable':
            print('Using trainable positional embeddings')
            if no_class_token_emb:
                # Positional embeddings for tokens only (no CLS)
                self.pos_embedding = nn.Parameter(
                    torch.randn(1, num_tokens, dim) * 0.02,
                    requires_grad=trainable_pos_emb
                )
            else:
                # Positional embeddings for tokens + CLS
                self.pos_embedding = nn.Parameter(
                    torch.randn(1, num_tokens + self.num_prefix_tokens, dim) * 0.02,
                    requires_grad=trainable_pos_emb
                )
                
        elif self.use_pe == 'sin-cos':
            print('Using Sin-Cos positional embeddings')
            if no_class_token_emb:
                self.pos_embedding = nn.Parameter(
                    torch.zeros(1, num_tokens, dim),
                    requires_grad=False
                )
            else:
                self.pos_embedding = nn.Parameter(
                    torch.zeros(1, num_tokens + self.num_prefix_tokens, dim),
                    requires_grad=False
                )
            self._init_pos_em()
        else:
            # No positional embeddings
            self.pos_embedding = None
        
        # ====================================================================
        # CLS TOKEN
        # ====================================================================
        if weights_layers_init:
            print('Using initialization from Timm repo')
            self.cls_token = nn.Parameter(torch.zeros(1, 1, dim)) if use_class_token else None
            self._init_weights_class()
            self._init_weights()
        else:
            self.cls_token = nn.Parameter(torch.randn(1, 1, dim)) if use_class_token else None

    def _init_weights_class(self):
        """Initialize weights using timm-style"""
        if self.pos_embedding is not None:
            trunc_normal_(self.pos_embedding, std=0.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        
        # Initialize nn.Linear layers
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialize linear layer weights"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def _init_pos_em(self):
        """Initialize sinusoidal positional embeddings"""
        num_positions = self.num_tokens if self.no_class_token_emb else self.num_tokens + 1
        pos_embed = get_1d_sincos_pos_embed_from_grid(
            self.pos_embedding.shape[-1],
            np.arange(num_positions, dtype=np.float32)
        )
        self.pos_embedding.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
    
    def add_pos_embed(self, x):
        """
        Add positional embeddings to input tokens
        
        Args:
            x: [batch, num_tokens, dim] input tokens
        
        Returns:
            x: [batch, num_tokens + (1 if cls), dim] with positional embeddings
        """
        b, n, _ = x.shape
        
        if self.pos_embedding is None:
            # No positional embeddings
            if self.use_class_token:
                cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
                x = torch.cat((cls_tokens, x), dim=1)
            return x
        
        if self.no_class_token_emb:
            # Add pos embeddings to tokens, then prepend CLS
            x = x + self.pos_embedding[:, :n, :]
            if self.use_class_token:
                cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
                x = torch.cat((cls_tokens, x), dim=1)
        else:
            # Prepend CLS, then add pos embeddings to all
            if self.use_class_token:
                cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
                x = torch.cat((cls_tokens, x), dim=1)
            x = x + self.pos_embedding[:, :x.shape[1], :]
        
        return x

    def forward(self, tokens, confounds=None):
        """
        Forward pass
        
        Args:
            tokens: [batch, num_tokens, token_dim] - Input tokens
            confounds: [batch, 1] - Optional confounds
        
        Returns:
            output: [batch, num_classes] - Model predictions
        """
        
        # Project input tokens to hidden dimension
        # Input: [batch, num_tokens, token_dim]
        # Output: [batch, num_tokens, dim]
        x = self.to_patch_embedding(tokens)
        
        b, n, _ = x.shape
        
        # Add positional embeddings and CLS token if used
        x = self.add_pos_embed(x)
        
        # Add confounds if provided
        if self.use_confounds and (confounds is not None):
            confounds = self.proj_confound(confounds.view(-1, 1))
            num_positions = n + 1 if self.use_class_token else n
            confounds = repeat(confounds, 'b d -> b n d', n=num_positions)
            x = x + confounds
        
        # Apply dropout
        x = self.dropout(x)
        
        # Apply transformer
        x = self.transformer(x)
        
        # Pool output
        if self.pool == 'mean':
            x = x.mean(dim=1)
        else:  # 'cls'
            x = x[:, 0]
        
        # Apply output head
        x = self.to_latent(x)
        output = self.mlp_head(x)
        
        return output
    
    def get_embedding(self, tokens, confounds=None):
        """
        Extract embeddings without classification head
        
        Args:
            tokens: [batch, num_tokens, token_dim] - Input tokens
            confounds: [batch, 1] - Optional confounds
        
        Returns:
            embedding: [batch, dim] - Learned embeddings
        """
        
        # Project tokens
        x = self.to_patch_embedding(tokens)
        
        b, n, _ = x.shape
        
        # Add positional embeddings and CLS token
        x = self.add_pos_embed(x)
        
        # Add confounds if provided
        if self.use_confounds and (confounds is not None):
            confounds = self.proj_confound(confounds.view(-1, 1))
            num_positions = n + 1 if self.use_class_token else n
            confounds = repeat(confounds, 'b d -> b n d', n=num_positions)
            x = x + confounds
        
        # Apply dropout
        x = self.dropout(x)
        
        # Apply transformer
        x = self.transformer(x)
        
        # Pool output
        if self.pool == 'mean':
            embedding = x.mean(dim=1)
        else:  # 'cls'
            embedding = x[:, 0]
        
        return embedding


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_sit_usage():
    """Example of how to use the modified SiT model"""
    
    # Configuration
    batch_size = 32
    num_tokens = 320
    token_dim = 768
    
    # Create dummy tokenized fMRI data
    tokens = torch.randn(batch_size, num_tokens, token_dim)
    
    print(f"Input shape: {tokens.shape}")
    print(f"  Batch size: {batch_size}")
    print(f"  Num tokens: {num_tokens}")
    print(f"  Token dim: {token_dim}")
    
    # Initialize model
    model = SiT(
        dim=384,                # Hidden dimension
        depth=12,               # Number of layers
        heads=6,                # Number of attention heads
        num_tokens=num_tokens,  # Number of input tokens
        token_dim=token_dim,    # Input token dimension
        num_classes=10,         # Number of output classes
        pool='cls',             # Use CLS token pooling
        use_pe='sin-cos',       # Use sinusoidal positional embeddings
        use_class_token=True,   # Use CLS token
        dropout=0.1
    )
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Forward pass
    output = model(tokens)
    print(f"\nOutput shape: {output.shape}")  # [32, 10]
    
    # Extract embeddings (without classification head)
    embeddings = model.get_embedding(tokens)
    print(f"Embedding shape: {embeddings.shape}")  # [32, 384]
    
    return model


def example_with_confounds():
    """Example with confounds"""
    
    batch_size = 16
    num_tokens = 20
    token_dim = 512
    
    tokens = torch.randn(batch_size, num_tokens, token_dim)
    confounds = torch.randn(batch_size, 1)  # Age, motion, etc.
    
    model = SiT(
        dim=256,
        depth=8,
        heads=4,
        num_tokens=num_tokens,
        token_dim=token_dim,
        num_classes=1,
        use_confounds=True,  # Enable confounds
        use_pe='trainable',  # Use learnable positional embeddings
    )
    
    output = model(tokens, confounds=confounds)
    print(f"Output with confounds: {output.shape}")
    
    return model


def example_bottleneck():
    """Example with bottleneck projection"""
    
    batch_size = 8
    num_tokens = 50
    token_dim = 2048  # Large input dimension
    
    tokens = torch.randn(batch_size, num_tokens, token_dim)
    
    model = SiT(
        dim=384,
        depth=12,
        heads=6,
        num_tokens=num_tokens,
        token_dim=token_dim,
        num_classes=5,
        use_bottleneck=True,  # Use bottleneck to reduce dimension
        bottleneck_dropout=0.1,
        pool='mean'  # Use mean pooling instead of CLS
    )
    
    output = model(tokens)
    print(f"Output with bottleneck: {output.shape}")
    
    return model


if __name__ == "__main__":
    print("=" * 80)
    print("SiT Model for Tokenized Input - Examples")
    print("=" * 80)
    
    print("\n1. Basic Usage:")
    print("-" * 80)
    model1 = example_sit_usage()
    
    print("\n2. With Confounds:")
    print("-" * 80)
    model2 = example_with_confounds()
    
    print("\n3. With Bottleneck:")
    print("-" * 80)
    model3 = example_bottleneck()
    
    print("\n" + "=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)