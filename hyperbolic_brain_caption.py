"""
Brain-to-Caption via Hyperbolic Alignment
================================================================================
- fMRI → (257, 1024) predicted CLIP features
- Alignment at CLIP features level
- Uses Stage 1 trained components (hyp_proj_image, image_encoder, llm_proj) frozen

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional, Tuple, Dict, List
from einops import rearrange, repeat
import warnings
from sit import SiT
import gc

warnings.filterwarnings('ignore')


# ============================================================================
# 1. HYPERBOLIC GEOMETRY UTILITIES
# ============================================================================

class HyperbolicOperations:
    """Poincaré ball operations with numerical stability"""
    
    def __init__(self, c: float = 0.1, epsilon: float = 1e-5):
        self.c = c
        self.epsilon = epsilon
        self.max_norm = 1.0 - epsilon
    
    def exp_map(self, u: torch.Tensor, base: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Exponential map: Tangent space (Euclid) -> Hyperbolic space"""
        if base is None:
            base = torch.zeros_like(u)
        
        u_norm = torch.clamp(u.norm(dim=-1, keepdim=True), min=self.epsilon)
        sqrt_c = torch.sqrt(torch.tensor(self.c, device=u.device))
        lambda_base = 2.0 / (1.0 - self.c * (base ** 2).sum(dim=-1, keepdim=True))
        
        direction = u / u_norm
        coef = torch.tanh(sqrt_c * lambda_base * u_norm / 2.0)
        
        z = base + coef * direction / sqrt_c
        
        # Project to valid region
        z_norm = z.norm(dim=-1, keepdim=True)
        z = torch.where(z_norm > self.max_norm, z / z_norm * self.max_norm, z)
        
        return z
    
    def log_map(self, z: torch.Tensor, base: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Logarithm map: Hyperbolic space -> Tangent space"""
        if base is None:
            base = torch.zeros_like(z)
        
        diff = z - base
        diff_norm = torch.clamp(diff.norm(dim=-1, keepdim=True), min=self.epsilon)
        
        sqrt_c = torch.sqrt(torch.tensor(self.c, device=z.device))
        lambda_base = 2.0 / (1.0 - self.c * (base ** 2).sum(dim=-1, keepdim=True))
        
        z_norm_sq = (z ** 2).sum(dim=-1, keepdim=True)
        atanh_arg = torch.clamp(sqrt_c * diff_norm, max=1.0 - self.epsilon)
        
        coef = 2.0 / (lambda_base * sqrt_c) * torch.atanh(atanh_arg)
        u = coef * diff / diff_norm
        
        return u
    
    def hyperbolic_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute distance in Poincaré ball"""
        diff = x - y
        diff_norm_sq = (diff ** 2).sum(dim=-1, keepdim=True)
        
        x_norm_sq = (x ** 2).sum(dim=-1, keepdim=True)
        y_norm_sq = (y ** 2).sum(dim=-1, keepdim=True)
        
        num = 2.0 * diff_norm_sq
        denom = (1.0 - self.c * x_norm_sq) * (1.0 - self.c * y_norm_sq)
        denom = torch.clamp(denom, min=self.epsilon)
        
        arcosh_arg = 1.0 + num / denom
        arcosh_arg = torch.clamp(arcosh_arg, min=1.0 + self.epsilon)
        
        distance = (1.0 / torch.sqrt(torch.tensor(self.c, device=x.device))) * torch.acosh(arcosh_arg)
        
        return distance.squeeze(-1)


# ============================================================================
# 2. PERCEIVER CROSS-ATTENTION
# ============================================================================

class PerceiverCrossAttention(nn.Module):
    """
    Perceiver-style cross-attention for token number transformation
    
    Query: learnable latent tokens [B, L, H]
    Key/Value: input tokens [B, P, D]
    Output: latent tokens [B, L, H]
    """
    
    def __init__(
        self,
        token_dim: int,
        latent_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = latent_dim // num_heads
        assert latent_dim % num_heads == 0, "latent_dim must be divisible by num_heads"
        
        self.scale = self.head_dim ** -0.5
        
        # Query projection (from latent)
        self.to_q = nn.Linear(latent_dim, latent_dim)
        
        # Key/Value projection (from input tokens)
        self.to_k = nn.Linear(token_dim, latent_dim)
        self.to_v = nn.Linear(token_dim, latent_dim)
        
        # Output projection
        self.to_out = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Dropout(dropout)
        )
        
        self.norm_latent = nn.LayerNorm(latent_dim)
        self.norm_tokens = nn.LayerNorm(token_dim)
        
    def forward(
        self,
        latents: torch.Tensor,
        tokens: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            latents: [B, L, H] - learnable latent queries
            tokens: [B, P, D] - input tokens (key/value)
            token_mask: [B, P] - mask for valid tokens (True = valid, False = padding)
        
        Returns:
            output: [B, L, H] - attended latent tokens
            attn: [B, num_heads, L, P] - attention weights
        """
        
        B, L, H = latents.shape
        _, P, D = tokens.shape
        
        # Normalize inputs
        latents_norm = self.norm_latent(latents)
        tokens_norm = self.norm_tokens(tokens)
        
        # Project to Q, K, V
        q = self.to_q(latents_norm)  # [B, L, H]
        k = self.to_k(tokens_norm)   # [B, P, H]
        v = self.to_v(tokens_norm)   # [B, P, H]
        
        # Reshape for multi-head attention
        q = rearrange(q, 'b l (h d) -> b h l d', h=self.num_heads)
        k = rearrange(k, 'b p (h d) -> b h p d', h=self.num_heads)
        v = rearrange(v, 'b p (h d) -> b h p d', h=self.num_heads)
        
        # Compute attention scores
        attn = torch.einsum('b h l d, b h p d -> b h l p', q, k) * self.scale
        
        # Apply mask if provided
        if token_mask is not None:
            # Expand mask for heads: [B, P] -> [B, 1, 1, P]
            mask = token_mask[:, None, None, :]
            attn = attn.masked_fill(~mask, float('-inf'))
        
        # Softmax and apply to values
        attn = F.softmax(attn, dim=-1)
        
        # Aggregate values
        out = torch.einsum('b h l p, b h p d -> b h l d', attn, v)
        out = rearrange(out, 'b h l d -> b l (h d)')
        
        # Output projection
        out = self.to_out(out)
        
        # Residual connection
        output = latents + out
        
        return output, attn


class PerceiverSelfAttention(nn.Module):
    """Self-attention on latent tokens"""
    
    def __init__(self, latent_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(latent_dim)
        
        # Feedforward
        self.ff = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 4, latent_dim),
            nn.Dropout(dropout)
        )
        self.norm_ff = nn.LayerNorm(latent_dim)
    
    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latents: [B, L, H]
        
        Returns:
            output: [B, L, H]
        """
        
        # Self-attention
        normed = self.norm(latents)
        attn_out, _ = self.attn(normed, normed, normed)
        latents = latents + attn_out
        
        # Feedforward
        latents = latents + self.ff(self.norm_ff(latents))
        
        return latents


# ============================================================================
# 3. PERCEIVER RESAMPLER (Encoder & Decoder)
# ============================================================================

class PerceiverResampler(nn.Module):
    """
    Perceiver resampler: transforms variable tokens to fixed latent tokens
    
    Used for:
    - Encoding: [B, P, D] -> [B, L, H]
    - Decoding: [B, L, H] -> [B, P, D]
    """
    
    def __init__(
        self,
        token_dim: int,
        latent_dim: int,
        num_latents: int,
        num_cross_attn_layers: int = 2,
        num_self_attn_layers: int = 2,
        num_heads: int = 8
    ):
        super().__init__()
        
        self.num_latents = num_latents
        
        # Learnable latent queries
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim) * 0.02)
        
        # Cross-attention layers
        self.cross_attn_layers = nn.ModuleList([
            PerceiverCrossAttention(token_dim, latent_dim, num_heads)
            for _ in range(num_cross_attn_layers)
        ])
        
        # Self-attention layers
        self.self_attn_layers = nn.ModuleList([
            PerceiverSelfAttention(latent_dim, num_heads)
            for _ in range(num_self_attn_layers)
        ])
        
    def forward(
        self,
        tokens: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
        external_latents: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Encode tokens to latent representation
        
        Args:
            tokens: [B, P, D] - input tokens
            token_mask: [B, P] - valid token mask
            external_latents: [B, L, H] - external latent queries (optional, e.g., from fine encoder)
        
        Returns:
            latents: [B, L, H] - latent representation
            cross_attn_maps: List of attention weights
        """
        
        B = tokens.shape[0]
        
        # Initialize latents - use external if provided, otherwise use learnable
        if external_latents is not None:
            latents = external_latents
        else:
            latents = repeat(self.latents, 'l h -> b l h', b=B)
        
        # Cross-attention: latents attend to input tokens
        cross_attn_maps = []
        for cross_attn in self.cross_attn_layers:
            latents, attn = cross_attn(latents, tokens, token_mask)
            cross_attn_maps.append(attn)
        
        # Self-attention: latents attend to each other
        for self_attn in self.self_attn_layers:
            latents = self_attn(latents)
        
        return latents, cross_attn_maps


class FineFeatureEncoder(nn.Module):
    """
    Fine visual cortex features → Perceiver latent queries
    
    Purpose:
        Initialize learnable latent queries from fMRI Perceiver
        using fine visual cortex information rather than random init.
    
    Input: (B, fine_dim) - fsaverage6 visual cortex vertices (e.g., 12535)
    Output: (B, num_latents, latent_dim) - Fine-informed latent queries (e.g., 257, 384)
    
    Key Design:
        - Residual connection with learnable base queries
        - fine_scale control fine info weight (learnable)
    """
    
    def __init__(
        self,
        fine_dim: int = 12535,
        latent_dim: int = 384,
        num_latents: int = 257,
        hidden_dim: int = 1024,
        dropout: float = 0.25,
    ):
        super().__init__()
        
        self.num_latents = num_latents
        self.latent_dim = latent_dim
        
        # Main encoder: fine features → latent queries
        self.encoder = nn.Sequential(
            nn.Linear(fine_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_latents * latent_dim),
        )
        
        # Normalization
        self.norm = nn.LayerNorm(latent_dim)
        
        # Learnable base queries (residual / fallback)
        self.base_latents = nn.Parameter(torch.randn(num_latents, latent_dim) * 0.02)
        
        # Learnable scale for fine contribution (starts small)
        self.fine_scale = nn.Parameter(torch.ones(1) * 0.3)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize with small weights for stability"""
        for m in self.encoder.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, fine_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fine_features: (B, fine_dim) - Fine visual cortex activations
        
        Returns:
            latent_queries: (B, num_latents, latent_dim) - Fine-informed latent queries
        """
        B = fine_features.shape[0]
        
        if self.training:
            fine_features = fine_features + torch.randn_like(fine_features) * 0.1       # Noise augmentation during training
        
        # Encode fine features
        encoded = self.encoder(fine_features)  # (B, num_latents * latent_dim)
        fine_latents = encoded.view(B, self.num_latents, self.latent_dim)
        
        # Residual combination: base + fine_scale * fine
        base = self.base_latents.unsqueeze(0).expand(B, -1, -1)
        latents = base + self.fine_scale * fine_latents
        
        return self.norm(latents)


# ============================================================================
# 4. PROJECTION TO HYPERBOLIC SPACE
# ============================================================================

class HyperbolicProjection(nn.Module):
    """Project latent tokens to hyperbolic space"""
    
    def __init__(self, latent_dim: int, hidden_dim: int = 512):
        super().__init__()
        
        self.proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Small initialization for stability
        for m in self.proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Project to tangent space (before exp map)
        
        Args:
            latents: [B, L, H]
        
        Returns:
            tangent: [B, L, H]
        """
        return self.proj(latents)


# ============================================================================
# 5. HYPERBOLIC LATENT MODEL (MAIN)
# ============================================================================

class HyperbolicBrainCaptionModel(nn.Module):
    """
    Two-Stage Brain-to-Caption Model
    
    Stage 1: Image → Perceiver → LLM Projection → Caption
    Stage 2: fMRI → Perceiver → Hyperbolic Alignment → Caption
    """
    
    def __init__(
        self,
        # Embedding dimensions
        image_token_dim: int = 1024,    # # CLIP ViT-L/14
        
        # fMRI Config
        fmri_raw_dim: int = 64,        # Raw input (B, 1000, 64)
        fmri_token_dim: int = 384,      # SiT Output or Precomputed (B, 1000, 384)
        
        # SiT Config
        sit_mode: str ='frozen',      # 'finetune', 'frozen', 'precomputed', 'disabled'
        sit_pretrained_path: Optional[str] = 'V:/XXXX/Project/Language_decoding/1.sit_pretraining/best_sit_pretrained.pth',  # or None
        sit_depth: int = 12,
        sit_heads: int = 6,
        
        # Latent space
        latent_dim: int = 768,
        num_latents: int = 16,      # K tokens (semantic compression)
        
        # fMRI Perceiver config (Early Alignment)
        fmri_num_latents: int = 257,    # Matches CLIP tokens for early alignment
        
        # Perceiver config
        num_cross_attn_layers: int = 2,
        num_self_attn_layers: int = 2,
        num_heads: int = 8,
        
        # Hyperbolic
        use_hyperbolic: bool = True,
        curvature: float = 0.1,
        
        # Fine Features (Stage 2)
        use_fine_features: bool = False,
        fine_dim: int = 12535,
        
        # LLM
        llm_dim: int = 4096         # LLaMA3 input dim 
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.num_latents = num_latents
        self.fmri_num_latents = fmri_num_latents
        
        # Hyperbolic operations
        self.use_hyperbolic = use_hyperbolic                    
        if use_hyperbolic:
            self.hyp_ops = HyperbolicOperations(c=curvature)
        else:
            self.hyp_ops = None                                 # Euclidean mode
        self.curvature = nn.Parameter(torch.tensor(curvature))
        
        # sit_mode
        """
        sit_mode:
            - 'finetune': Train SiT
            - 'frozen': SiT freeze
            - 'precomputed': No use SiT (Use precomputed embedding)
        """
        self.sit_mode = sit_mode
        
        # ====================================================================
        # 0. SiT Encoder Initialization (for Stage 2)
        # ====================================================================
        if sit_mode in ['frozen', 'finetune']:
            print(f"Initializing SiT Encoder (Mode: {sit_mode})")
            print(f"Input: {fmri_raw_dim} -> Output: {fmri_token_dim}")
            self.sit = SiT(
                dim=fmri_token_dim,
                depth=sit_depth,
                heads=sit_heads,
                num_tokens=1000,
                token_dim=fmri_raw_dim,
                num_classes=384,                # Dummy (Not use, set minimal)
                mlp_ratio=2,
                pool='cls',                     # Override in forward for sequence extraction
                use_pe='sin-cos',
                use_class_token=True,
                trainable_pos_emb=False
            )
            
            if sit_pretrained_path:
                print(f"Loading SiT weights from {sit_pretrained_path}")
                checkpoint = torch.load(sit_pretrained_path, map_location='cpu')
                
                # Checkpoint
                if 'model_state_dict' in checkpoint:
                    full_state_dict = checkpoint['model_state_dict']
                else:
                    full_state_dict = checkpoint
                
                # Extract only Encoder
                encoder_state_dict = {}
                for k, v in full_state_dict.items():
                    if k.startswith('encoder.'):
                        # Remove 'encoder.' prefix 
                        new_key = k.replace('encoder.', '', 1)
                        encoder_state_dict[new_key] = v
                
                # Load weight (ignore unexisted key using strict=False)
                missing_keys, unexpected_keys = self.sit.load_state_dict(encoder_state_dict, strict=False)
                
                # Detailed logging
                print(f"  ✓ Loaded {len(encoder_state_dict)} encoder keys")
                print(f"  Missing keys: {len(missing_keys)}")
                print(f"  Unexpected keys: {len(unexpected_keys)}")
                
                # Not use mlp_head
                if missing_keys:
                    mlp_head_keys = [k for k in missing_keys if 'mlp_head' in k]
                    if len(mlp_head_keys) == len(missing_keys):
                        print(f"  ✓ Only mlp_head keys missing (expected)")
                    else:
                        print(f"  ⚠️ Warning: {missing_keys}")
                        
                del checkpoint
                gc.collect()
                torch.cuda.empty_cache() 
            
            if sit_mode == 'frozen':
                print("Freezing SiT Encoder parameters.")
                for param in self.sit.parameters():
                    param.requires_grad = False
            else:
                print("SiT Encoder parameters are trainable (fine-tuning).")
            
        elif sit_mode == 'precomputed':
            print("SiT Encoder will not be used (precomputed embeddings expected).")
            self.sit = None
        
        elif sit_mode == 'disabled':
            print("⚠️ SiT DISABLED - Using Fine Features Only (Ablation)")
            self.sit = None
        
        # ==================================================================
        # Image PATH (Stage 1) - Will be frozen in Stage 2
        # ====================================================================
        
        # Image Perceiver: CLIP 257 tokens → K semantic tokens
        self.image_encoder = PerceiverResampler(
            token_dim=image_token_dim,                      # 1024
            latent_dim=image_token_dim,                     # 1024
            num_latents=num_latents,                        # 16
            num_cross_attn_layers=num_cross_attn_layers,
            num_self_attn_layers=num_self_attn_layers,
            num_heads=num_heads
        )
        
        # Hyperbolic projection for image
        self.hyp_proj_image = HyperbolicProjection(image_token_dim)
        
        # ====================================================================
        # fMRI PATH (Stage 2)
        # ====================================================================
        
        # fMRI Perceiver: SiT output (320, 384) → CLIP 257 tokens, alignment to image embedding
        self.fmri_encoder = PerceiverResampler(
            token_dim=fmri_token_dim,                       # 384 
            latent_dim=fmri_token_dim,                      # 384 
            num_latents=fmri_num_latents,                   # 257
            num_cross_attn_layers=num_cross_attn_layers,
            num_self_attn_layers=num_self_attn_layers,
            num_heads=num_heads
        )
        
        # Fine Feature Encoder (Stage 2) - produces latent queries from fine visual cortex
        self.use_fine_features = use_fine_features
        self.fine_dim = fine_dim
        if use_fine_features:
            self.fine_encoder = FineFeatureEncoder(
                fine_dim=fine_dim,
                latent_dim=fmri_token_dim,       # 384
                num_latents=fmri_num_latents,    # 257
                hidden_dim=1024,
                dropout=0.25
            )
            print(f"Fine Feature Encoder: ({fine_dim},) → ({fmri_num_latents}, {fmri_token_dim})")
        else:
            self.fine_encoder = None
        
        # Projection from fMRI space to Image space (384 → 1024)
        # Output: (257, 1024) - predicted CLIP features
        self.fmri_to_image_proj = nn.Sequential(
            nn.Linear(fmri_token_dim, latent_dim),     # 384 → 768
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim, image_token_dim),    # 768 → 1024
            nn.LayerNorm(image_token_dim)
        )
        
        # Hyperbolic projection for fMRI-image alignement
        self.hyp_proj_align  = HyperbolicProjection(image_token_dim)
        
        # ====================================================================
        # LLM PROJECTION (Stage 1 trained, Stage 2 frozen)
        # ====================================================================
        self.llm_proj = nn.Sequential(
            nn.Linear(image_token_dim, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, llm_dim),
            nn.LayerNorm(llm_dim)
        )
    
    def forward_sit_sequence(self, x):
        """SiT forward without pooling (returns sequence tokens)"""
        # 1. Patch Embedding
        x = self.sit.to_patch_embedding(x) # (B, 1000, 384)
        
        # 2. Add Pos Embedding (Returns B, 321, 384 including CLS)
        x = self.sit.add_pos_embed(x)      
        
        # 3. Transformer Layers
        x = self.sit.dropout(x)
        x = self.sit.transformer(x)        
        
        # 4. Remove CLS token (Index 0) to maintain sequence length 1000
        if self.sit.use_class_token:
            x = x[:, 1:, :]                # (B, 1000, 384)
            
        return x
    
    def forward(
        self,
        image_tokens: Optional[torch.Tensor] = None,    # (B, 257, 1024) - Stage 1
        fmri_data: Optional[torch.Tensor] = None,       # (B, 1000, 64/384) - Stage 2
        fine_features: Optional[torch.Tensor] = None,   # (B, fine_dim) - Fine visual cortex (Stage 2)
        stage: int = 1,                                 # 1 or 2
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with stage awareness
        
        Args:
            image_tokens: [B, P_img, D_img] - image tokens from CLIP-ViT
            fmri_tokens: [B, P_fmri, D_fmri] - fMRI tokens from SiT or surface patch raw fMRI data
            stage: Training stage (1 or 2)
            return_attention: Whether to return attention maps
        
        Returns:
            Dictionary of outputs
        """
        
        outputs = {}
        
        # ====================================================================
        # STAGE 1: Image → Semantic Tokens → LLM Input
        # ====================================================================
        if stage == 1 and image_tokens is not None:
            B = image_tokens.shape[0]
            
            # Encode image to latent (Euclidean)
            lat_image, attn_maps = self.image_encoder(image_tokens)  # [B, K, H==1024]
            
            # Project to hyperbolic space
            u_image = self.hyp_proj_image(lat_image)
            if self.use_hyperbolic:
                z_image = self.hyp_ops.exp_map(u_image)
            else:
                z_image = u_image
            
            outputs['lat_image'] = lat_image                        # [B, K, 1024] in IMAGE space
            outputs['z_image'] = z_image
            
            if return_attention:
                outputs['attn_image'] = torch.stack(attn_maps, dim=1)
            
            # Project to LLM space
            llm_input = self.llm_proj(lat_image)  # [B, K, llm_dim==4096]
            outputs['llm_input'] = llm_input
        
        
        # ====================================================================
        # STAGE 2: fMRI → Predicted CLIP → Frozen Pipeline → Caption
        # ====================================================================
        elif stage == 2 and fmri_data is not None:
            B = fmri_data.shape[0]
        
            if self.sit_mode == 'disabled':
                # Only Fine features
                if self.use_fine_features and self.fine_encoder is not None and fine_features is not None:
                    fine_latents = self.fine_encoder(fine_features)  # (B, 257, 384)
                    outputs['fine_latents'] = fine_latents
                    
                    # Option A: Using fine_latents directly lat_fmri (Pass fmri_encoder)
                    lat_fmri = fine_latents
                    
                    # Option B: Self-attention 
                    # lat_fmri, attn_maps = self.fmri_encoder(fine_latents, external_latents=fine_latents)
                    
                else:
                    raise ValueError("sit_mode='disabled' requires use_fine_features=True")
                
                outputs['lat_fmri'] = lat_fmri
            
            else:
                # Prepare fMRI embeddings
                if self.sit_mode == 'precomputed':
                    fmri_tokens = fmri_data  # (B, 1000, 384)
                else:
                    fmri_tokens = self.forward_sit_sequence(fmri_data)  # (B, 1000, 64) → (B, 1000, 384)
                
                outputs['fmri_tokens'] = fmri_tokens                    # fMRI embedding (B, 1000, 384)
                
                # Fine Feature Processing - create fine-informed latent queries
                fine_latents = None
                if self.use_fine_features and self.fine_encoder is not None and fine_features is not None:
                    fine_latents = self.fine_encoder(fine_features)     # (B, 257, 384)
                    outputs['fine_latents'] = fine_latents
                
                # STEP 1: Encode fMRI to CLIP image tokens (matches CLIP)
                # If fine_latents available, use as queries instead of learnable latents
                lat_fmri, attn_maps = self.fmri_encoder(fmri_tokens, external_latents=fine_latents)    # [B, 257, 384]
                outputs['lat_fmri'] = lat_fmri                          # [B, 257, 384] in fMRI space
                
                if return_attention:
                    outputs['attn_fmri'] = torch.stack(attn_maps, dim=1) # [B, Layers, Heads, Latents, Tokens]
                
            # STEP 2: Project from fMRI to predicted CLIP features    (B, 257, 384) → (B, 257, 1024)
            lat_image_pred  = self.fmri_to_image_proj(lat_fmri)     # [B, 257, 1024]
            outputs['lat_image_pred'] = lat_image_pred              # [B, 257, 1024] predicted CLIP features
            
            # STEP 3: Project to hyperbolic space for alignment fmri and image (CLIP output)
            u_image_pred  = self.hyp_proj_align(lat_image_pred)
            if self.use_hyperbolic:
                z_image_pred  = self.hyp_ops.exp_map(u_image_pred)
            else:
                z_image_pred  = u_image_pred                      # Euclidean mode
            
            outputs['z_image_pred'] = z_image_pred                  # [B, 257, 1024] in hyperbolic space            
            
            # STEP 4: Pass through frozen Image Perceiver → LLM Proj  (B, 257, 1024) → (B, K, 1024) semantic tokens
            # This generates the final semantic tokens for LLaMA
            
            # Use predicted CLIP features as input to frozen Image Perceiver
            lat_semantic, attn_maps_image = self.image_encoder(lat_image_pred)  # [B, K, 1024]
            
            outputs['lat_semantic'] = lat_semantic        # [B, K, 1024] semantic tokens
            
            # ============== Get image perceiver attention maps ==============
            if return_attention:                                            
                outputs['attn_image'] = torch.stack(attn_maps_image, dim=1) # [B, Layers, Heads, Latents, Tokens]
            
            
            # STEP 5: Project to LLM space via frozen llm_proj
            llm_input = self.llm_proj(lat_semantic)   # [B, K, 4096]
            
            outputs['llm_input'] = llm_input
            
        return outputs


# ============================================================================
# 6. DATASET (Simplified for Two-Stage)
# ============================================================================

class NSDCaptionDataset(Dataset):
    """
    Dataset for NSD with multiple captions per fMRI sample
    
    Each fMRI sample has K captions (default: 5)
    
    Dataset for two-stage training
    
    Stage 1: Returns image tokens + 1 random caption
    Stage 2: Returns fMRI data + 1 random caption + target image embedding
    """
    
    def __init__(
        self,
        image_paths,                   
        fmri_paths,                      
        caption_paths,             
        fine_paths=None,               # Fine visual cortex paths (optional)
        stage: int = 1,
        llama_tokenizer=None,            # LLaMA tokenizer for captions tokens
        max_caption_length: int = 50,    # max caption length
        ID_list=None,
        use_all_captions: bool = False,
    ):
        self.image_paths = image_paths
        self.fmri_paths = fmri_paths
        self.caption_paths = caption_paths
        self.fine_paths = fine_paths
        self.stage = stage
        self.llama_tokenizer = llama_tokenizer
        self.max_caption_length = max_caption_length
        self.ID_list = ID_list
        self.use_all_captions = use_all_captions
        
        self.num_samples = len(fmri_paths)
        
        if use_all_captions:
            self.effective_samples = self.num_samples * 5
        else:
            self.effective_samples = self.num_samples
    
    def __len__(self):
        return self.effective_samples
    
    def __getitem__(self, idx):
        item = {}
        
        if self.use_all_captions:
            sample_idx = idx // 5                   # Each sample has 5 captions
            caption_idx = idx % 5
        else:
            sample_idx = idx
            caption_idx = np.random.randint(0, 5)
        
        # Load fmri, image, caption
        image_data_mmap = np.load(self.image_paths[sample_idx], mmap_mode='r')          # Memory-mapped
        image_data = image_data_mmap.copy()
        if image_data.ndim == 3 and image_data.shape[0] == 1:
            image_data = image_data.squeeze(0)                              # remove only first dimension
        image_tokens = torch.from_numpy(image_data).float()                         # [P_img=257, D_img=1024]
        
        fmri_data_mmap = np.load(self.fmri_paths[sample_idx], mmap_mode='r')  
        fmri_data = fmri_data_mmap.copy()
        fmri_data = torch.from_numpy(fmri_data).float()                             # [P_fmri=320, D_fmri=64/384], raw for fine-finetuning or precomputed embeddings
        
        caption_data_mmap = np.load(self.caption_paths[sample_idx], mmap_mode='r')              # List of [K=5] caption strings
        caption_data = caption_data_mmap.copy()
        caption_text = caption_data[caption_idx]
        
        if self.ID_list is not None:
            item['ID'] = self.ID_list[sample_idx]
        
        if self.stage == 1:
            # Stage 1: Image + Caption
            item['image_tokens'] = image_tokens
        elif self.stage == 2:
            # Stage 2: fMRI + Caption + Target Image Embedding
            item['fmri_data'] = fmri_data
            item['target_image_tokens'] = image_tokens  # For alignment target
            
            # Load fine features if available
            if self.fine_paths is not None:
                fine_data_mmap = np.load(self.fine_paths[sample_idx], mmap_mode='r')
                fine_data = fine_data_mmap.copy()
                item['fine_features'] = torch.from_numpy(fine_data).float()  # (fine_dim,)
        
        
        # Tokenize captions for LLaMA if tokenizer is provided
        if self.llama_tokenizer is not None:
            caption_with_eos = str(caption_text).strip() + self.llama_tokenizer.eos_token  # add <EOS> token
            
            tokens = self.llama_tokenizer(
                caption_with_eos,
                max_length=self.max_caption_length,
                padding='max_length',
                truncation=True,
                add_special_tokens=False,
                return_tensors='pt'
            )
                
            item['caption_tokens'] = tokens['input_ids'].squeeze(0)
            item['caption_mask'] = tokens['attention_mask'].squeeze(0)
        item['caption_text'] = caption_text
        item['captions'] = caption_data
        item['sample_idx'] = sample_idx                                 # for contrastive learning identify positive pair
        
        return item


def collate_caption_batch(batch):
    """
    Custom collate function for caption dataset
    """
    
    collated = {'caption_texts': [item['caption_text'] for item in batch],
                'sample_idx': torch.tensor([item['sample_idx'] for item in batch])
                }
    
    # Stack single-valued items
    if 'image_tokens' in batch[0]:
        collated['image_tokens'] = torch.stack([item['image_tokens'] for item in batch])
    
    if 'fmri_data' in batch[0]:
        collated['fmri_data'] = torch.stack([item['fmri_data'] for item in batch])
    
    if 'target_image_tokens' in batch[0]:
        collated['target_image_tokens'] = torch.stack([item['target_image_tokens'] for item in batch])
    
    if 'fine_features' in batch[0]:
        collated['fine_features'] = torch.stack([item['fine_features'] for item in batch])
    
    if 'caption_tokens' in batch[0]:
        collated['caption_tokens'] = torch.stack([item['caption_tokens'] for item in batch])
        collated['caption_mask'] = torch.stack([item['caption_mask'] for item in batch])
    
    if 'captions' in batch[0]:
        collated['captions'] = [item['captions'] for item in batch]
    
    if 'ID' in batch[0]:
        collated['ID'] = [item['ID'] for item in batch]
    
    return collated


# ============================================================================
# 8. EXAMPLE USAGE (Not use)
# ============================================================================

def example_two_stage_model():
    """Example of using the two-stage model"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Configuration
    batch_size = 4
    
    # ====================================================================
    # Stage 1: Image → Caption
    # ====================================================================
    print("\n" + "="*60)
    print("STAGE 1: Image → Caption")
    print("="*60)
    
    # Create dummy data
    # Image: CLIP-ViT tokens (1 CLS + 256 patches = 257 tokens)
    image_tokens = torch.randn(batch_size, 257, 1024)
    
    # Initialize model
    model = HyperbolicBrainCaptionModel(
        image_token_dim=1024,
        text_token_dim=768,
        fmri_raw_dim = 64,        # Raw input (B, 320, 64)
        fmri_token_dim = 384,      # SiT Output or Precomputed (B, 320, 384)
        latent_dim=768,
        num_latents=16,
        use_hyperbolic=True,
        sit_mode ='frozen',      # 'finetune', 'frozen', 'precomputed'
        sit_pretrained_path = 'V:/XXXX/Project/Language_decoding/1.sit_pretraining/best_sit_pretrained.pth',  # or None
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    
    # Forward pass
    image_tokens = image_tokens.to(device)
    
    outputs_s1 = model(
        image_tokens=image_tokens,
        stage=1,
        return_attention=True
    )
    
    # Check outputs
    print("\nStage 1 Outputs:")
    print(f"  lat_image: {outputs_s1['lat_image'].shape}")      # (B, K, H)
    print(f"  z_image: {outputs_s1['z_image'].shape}")          # (B, K, H)
    print(f"  llm_input: {outputs_s1['llm_input'].shape}")      # (B, K, llm_dim)
    if 'attn_image' in outputs_s1:
        print(f"  attn_image: {outputs_s1['attn_image'].shape}")  # (B, Layers, Heads, K, 257)
    
    # ====================================================================
    # Stage 2: fMRI → Image Embedding → Caption
    # ====================================================================
    print("\n" + "="*60)
    print("STAGE 2: fMRI → Caption")
    print("="*60)
    
    # Create dummy fMRI data
    fmri_data = torch.randn(batch_size, 1000, 64).to(device)
    
    # Forward pass (Stage 2)
    outputs_s2 = model(
        fmri_data=fmri_data,
        stage=2,
        return_attention=True
    )
    
    print("\nStage 2 Outputs:")
    print(f"  fmri_tokens: {outputs_s2['fmri_tokens'].shape}")                 # (B, 320, 384)
    print(f"  lat_fmri: {outputs_s2['lat_fmri'].shape}")                       # (B, K, 384)
    print(f"  z_fmri: {outputs_s2['z_fmri'].shape}")                           # (B, K, 384)
    print(f"  lat_image_pred: {outputs_s2['lat_image_pred'].shape}")           # (B, K, 1024) predicted
    print(f"  z_image_pred: {outputs_s2['z_image_pred'].shape}")               # (B, K, 1024) in hyperbolic
    print(f"  lat_fmri_recon: {outputs_s2['lat_fmri_recon'].shape}")           # (B, K, 384) reconstructed
    print(f"  llm_input: {outputs_s2['llm_input'].shape}")                     # (B, K, llm_dim)
    if 'attn_fmri' in outputs_s2:
        print(f"  attn_fmri: {outputs_s2['attn_fmri'].shape}")           # (B, Layers, Heads, K, 320)
    
    # ====================================================================
    # Compute Alignment Loss (Example)
    # ====================================================================
    print("\n" + "="*60)
    print("Alignment Check")
    print("="*60)
    
    # Compare predicted image (from fMRI) with GT image
    # Compute distance between z_image_pred and z_image
    if model.use_hyperbolic:
        distances = []
        for k in range(model.num_latents):
            dist_k = model.hyp_ops.hyperbolic_distance(
                outputs_s2['z_image_pred'][:, k, :],  # Predicted image (from fMRI)
                outputs_s1['z_image'][:, k, :]        # GT image
            )
            distances.append(dist_k.mean().item())
        
        print(f"\nToken-wise Hyperbolic Distances (Predicted vs GT Image):")
        for k, dist in enumerate(distances):
            print(f"  Token {k}: {dist:.4f}")
        print(f"  Mean: {np.mean(distances):.4f}")
    
    # Check cycle consistency
    print(f"\nCycle Consistency Check:")
    cycle_error = F.mse_loss(outputs_s2['lat_fmri_recon'], outputs_s2['lat_fmri']).item()
    print(f"  MSE(fMRI_recon, fMRI_original): {cycle_error:.6f}")
    
    print("\n" + "="*60)
    print("Example Complete!")
    print("="*60)
    
    return model, outputs_s1, outputs_s2


if __name__ == "__main__":
    model, outputs_s1, outputs_s2 = example_two_stage_model()