"""
losses.py
Loss Functions for Two-Stage Framework
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple

from hyperbolic_brain_caption import HyperbolicOperations

# ============================================================================
# CONFIGURATION
# ============================================================================

class StageConfig:
    """Configuration for two-stage training"""
    
    def __init__(
        self,
        stage: int = 1,
        use_hyperbolic: bool = True,
        use_cciea: bool = False,
        use_contrastive: bool = False,      # Hyperbolic contrastive loss
        use_hyperbolic_infonce: bool = False,
    ):
        self.stage = stage
        self.use_hyperbolic = use_hyperbolic
        self.use_cciea = use_cciea
        self.use_contrastive = use_contrastive
        self.use_hyperbolic_infonce = use_hyperbolic_infonce
    
    @staticmethod
    def get_stage1():
        """Stage 1: Image-to-Caption (Vision-Language Grounding)"""
        return StageConfig(stage=1, use_hyperbolic=False, use_cciea=False)
    
    @staticmethod
    def get_stage2_early_mse():
        """Stage 2: Early Alignment with MSE + CCIEA"""
        return StageConfig(
            stage=2, 
            use_hyperbolic=False,
            use_cciea=True,
            use_contrastive=False
        )
    
    @staticmethod
    def get_stage2_early_hyperbolic():
        """Stage 2: Early Alignment with Hyperbolic Distance + CCIEA"""
        return StageConfig(
            stage=2, 
            use_hyperbolic=True,
            use_cciea=True,
            use_contrastive=False
        )
    
    @staticmethod
    def get_stage2_early_contrastive():
        """Stage 2: Early Alignment with Hyperbolic Contrastive + CCIEA"""
        return StageConfig(
            stage=2, 
            use_hyperbolic=True,
            use_cciea=True,
            use_contrastive=True
        )
    
    @staticmethod
    def get_stage2_hyperbolic_infonce():
        """Stage 2: Hyperbolic InfoNCE + CCIEA (Proposed Method)"""
        return StageConfig(
            stage=2, 
            use_hyperbolic=True,
            use_cciea=True,
            use_contrastive=False,
            use_hyperbolic_infonce=True
        )
    
    def __str__(self):
        if self.stage == 1:
            return "stage1_image_caption"
        else:
            parts = ["stage2_early"]
            if self.use_contrastive:
                parts.append("contrastive")
            elif self.use_hyperbolic:
                parts.append("hyperbolic")
            else:
                parts.append("mse")
            if self.use_cciea:
                parts.append("cciea")
            return "_".join(parts)


# ============================================================================
# LOSS CLASSES
# ============================================================================

class HyperbolicAlignmentLoss(nn.Module):
    """
    Token-wise hyperbolic distance loss for fMRI-Image alignment
    Aligns z_image_pred (from fMRI) with z_image (GT)
    """
    
    def __init__(self, curvature: float = 0.1, reduction: str = 'mean'):
        super().__init__()
        self.hyp_ops = HyperbolicOperations(c=curvature)
        self.reduction = reduction
    
    def forward(
        self,
        z_image_pred: torch.Tensor,     # [B, K, H] - Predicted image (from fMRI)
        z_image: torch.Tensor           # [B, K, H] - GT image (from Image Perceiver)
    ) -> torch.Tensor:
        """
        Compute token-wise hyperbolic distance
        
        Args:
            z_image_pred: Predicted image latent in hyperbolic space (from fMRI)
            z_image: Ground truth image latent in hyperbolic space
        
        Returns:
            loss: Mean hyperbolic distance across tokens
        """
        
        B, K, H = z_image_pred.shape
        
        # Compute pairwise distance for each token
        # [B, K] - distance for each of K semantic tokens
        distances = []
        for k in range(K):
            dist_k = self.hyp_ops.hyperbolic_distance(
                z_image_pred[:, k, :],  # [B, H] - predicted
                z_image[:, k, :]        # [B, H] - GT
            )  # [B]
            distances.append(dist_k)
        
        distances = torch.stack(distances, dim=1)  # [B, K]
        
        # Reduce
        if self.reduction == 'mean':
            loss = distances.mean()
        elif self.reduction == 'sum':
            loss = distances.sum()
        else:
            loss = distances
        
        return loss

class HyperbolicContrastiveLoss(nn.Module):
    """
    Hyperbolic Contrastive Loss for fMRI-Image alignment
    
    Combines hyperbolic distance with contrastive learning:
    - Pull matched fMRI-Image pairs together
    - Push unmatched pairs apart
    
    Uses InfoNCE-style contrastive loss in hyperbolic space.
    """
    
    def __init__(self, curvature: float = 0.1, temperature: float = 0.1, reduction: str = 'mean'):
        super().__init__()
        self.hyp_ops = HyperbolicOperations(c=curvature)
        self.temperature = temperature
        self.reduction = reduction
    
    def forward(
        self,
        z_fmri: torch.Tensor,     # [B, K, H] - fMRI embeddings in hyperbolic space
        z_image: torch.Tensor,    # [B, K, H] - Image embeddings in hyperbolic space
        sample_idx
    ) -> torch.Tensor:
        """
        Compute hyperbolic contrastive loss
        
        Args:
            z_fmri: fMRI embeddings in hyperbolic space [B, K, H]
            z_image: Image embeddings in hyperbolic space [B, K, H]
            sample_idx: (B,) Each sample original index (same index = same image different caption)
        
        Returns:
            loss: Contrastive loss (bidirectional)
        """
        
        B, K, H = z_fmri.shape
        
        # Pool tokens to get sample-level representations
        # Option 1: Mean pooling across tokens
        z_fmri_pooled = z_fmri.mean(dim=1)    # [B, H]
        z_image_pooled = z_image.mean(dim=1)  # [B, H]
        
        # Compute pairwise hyperbolic distances [B, B]
        # distances[i, j] = distance between fMRI_i and Image_j
        distances = torch.zeros(B, B, device=z_fmri.device)
        for i in range(B):
            for j in range(B):
                distances[i, j] = self.hyp_ops.hyperbolic_distance(
                    z_fmri_pooled[i:i+1],
                    z_image_pooled[j:j+1]
                ).squeeze()
        
        # Convert distances to similarities (negative distance / temperature)
        # Smaller distance = higher similarity
        similarities = -distances / self.temperature
        
        # sample_idx = positive pair
        # positive_mask[i, j] = True if sample_idx[i] == sample_idx[j]
        positive_mask = (sample_idx.unsqueeze(0) == sample_idx.unsqueeze(1))  # (B, B)
        
        # Multi-positive contrastive loss (All captions in same sample are positive)
        # exp(sim) / (sum of all exp(sim))
        # but with multiple positives
        
        exp_sim = torch.exp(similarities)
        
        # For each row, sum of positive exp_sim
        positive_sum = (exp_sim * positive_mask.float()).sum(dim=1)  # (B,)
        
        # Total sum per row
        total_sum = exp_sim.sum(dim=1)  # (B,)
        
        # Loss: -log(positive_sum / total_sum)
        loss = -torch.log(positive_sum / total_sum + 1e-8).mean()
        
        return loss


class HyperbolicInfoNCELossVectorized(nn.Module):
    """
    Hyperbolic InfoNCE Loss - Vectorized Version
    Not use for loop
    """
    
    def __init__(
        self, 
        curvature: float = 0.1, 
        temperature: float = 0.1,
        learnable_temp: bool = False
    ):
        super().__init__()
        self.c = curvature
        self.epsilon = 1e-5
        
        if learnable_temp:
            self.log_temp = nn.Parameter(torch.tensor(np.log(temperature)))
        else:
            self.register_buffer('log_temp', torch.tensor(np.log(temperature)))
    
    @property
    def temperature(self):
        return torch.exp(self.log_temp)
    
    def pairwise_hyperbolic_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Vectorized pairwise hyperbolic distance
        
        Args:
            x: (B1, H)
            y: (B2, H)
        
        Returns:
            distances: (B1, B2)
        """
        B1, H = x.shape
        B2 = y.shape[0]
        
        # Expand for pairwise computation
        x_exp = x.unsqueeze(1).expand(B1, B2, H)  # (B1, B2, H)
        y_exp = y.unsqueeze(0).expand(B1, B2, H)  # (B1, B2, H)
        
        # ||x - y||^2
        diff = x_exp - y_exp
        diff_norm_sq = (diff ** 2).sum(dim=-1)  # (B1, B2)
        
        # ||x||^2, ||y||^2
        x_norm_sq = (x ** 2).sum(dim=-1, keepdim=True)  # (B1, 1)
        y_norm_sq = (y ** 2).sum(dim=-1).unsqueeze(0)   # (1, B2)
        
        # Hyperbolic distance formula
        num = 2.0 * diff_norm_sq
        denom = (1.0 - self.c * x_norm_sq) * (1.0 - self.c * y_norm_sq)
        denom = torch.clamp(denom, min=self.epsilon)
        
        arcosh_arg = 1.0 + num / denom
        arcosh_arg = torch.clamp(arcosh_arg, min=1.0 + self.epsilon)
        
        distances = (1.0 / np.sqrt(self.c)) * torch.acosh(arcosh_arg)
        
        # Clamp for stability
        distances = torch.clamp(distances, min=0.0, max=10.0)
        
        return distances
    
    def forward(
        self, 
        z_fmri: torch.Tensor,
        z_image: torch.Tensor,
        sample_idx: torch.Tensor
    ) -> torch.Tensor:
        
        # Pool tokens
        if z_fmri.dim() == 3:
            z_fmri_pooled = z_fmri.mean(dim=1)
            z_image_pooled = z_image.mean(dim=1)
        else:
            z_fmri_pooled = z_fmri
            z_image_pooled = z_image
        
        B = z_fmri_pooled.shape[0]
        
        # Pairwise hyperbolic distances
        distances = self.pairwise_hyperbolic_distance(z_fmri_pooled, z_image_pooled)
        
        # Convert to logits (negative distance / temperature)
        logits = -distances / self.temperature
        
        # Multi-positive mask
        positive_mask = (sample_idx.unsqueeze(1) == sample_idx.unsqueeze(0))
        
        # Bidirectional InfoNCE
        loss_f2i = self._infonce_loss(logits, positive_mask)
        loss_i2f = self._infonce_loss(logits.t(), positive_mask.t())
        
        return (loss_f2i + loss_i2f) / 2.0
    
    def _infonce_loss(self, logits, positive_mask):
        logits_max = logits.max(dim=1, keepdim=True)[0].detach()
        exp_logits = torch.exp(logits - logits_max)
        
        positive_sum = (exp_logits * positive_mask.float()).sum(dim=1)
        total_sum = exp_logits.sum(dim=1)
        
        loss = -torch.log(positive_sum + 1e-8) + torch.log(total_sum + 1e-8)
        return loss.mean()


class CCIEALoss(nn.Module):
    """
    Caption-Conditioned Image Embedding Alignment (CCIEA) Loss
    
    Aligns fMRI representation in the caption-induced probability space
    of a frozen LLM, rather than in raw embedding space.
    
    Key idea:
        P_LLaMA(y | z_fmri) ≈ P_LLaMA(y | z_img)
    
    The fMRI embedding is "good" if LLaMA produces the same caption
    distribution as when given the image embedding.
    
    This is more powerful than MSE/contrastive because:
    1. Aligns in semantic space, not geometric space
    2. Training objective ≈ Evaluation metric (caption quality)
    3. Respects LLM's decision boundaries
    """
    
    def __init__(
        self,
        temperature: float = 1.0,
        reduction: str = 'batchmean'
    ):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.kl_loss = nn.KLDivLoss(reduction=reduction)
    
    def forward(
        self,
        logits_fmri: torch.Tensor,   # [B, T, V] - fMRI path logits
        logits_img: torch.Tensor,    # [B, T, V] - Image path logits (teacher, detached)
        caption_mask: Optional[torch.Tensor] = None  # [B, T] - valid token mask
    ) -> torch.Tensor:
        """
        Compute KL divergence between image and fMRI caption distributions
        
        Args:
            logits_fmri: Logits from fMRI path (student) - gradients flow here
            logits_img: Logits from image path (teacher, should be detached)
            caption_mask: Mask for valid caption tokens (1=valid, 0=pad)
        
        Returns:
            loss: KL divergence loss
        """
        
        # Apply temperature scaling
        logits_fmri_scaled = logits_fmri / self.temperature
        logits_img_scaled = logits_img / self.temperature
        
        # Convert to log probabilities (fMRI) and probabilities (image)
        log_p_fmri = F.log_softmax(logits_fmri_scaled, dim=-1)  # [B, T, V]
        p_img = F.softmax(logits_img_scaled, dim=-1)            # [B, T, V]
        
        # Apply mask if provided
        if caption_mask is not None:
            # Flatten for masked selection
            B, T, V = log_p_fmri.shape
            mask_flat = caption_mask.view(-1).bool()  # [B*T]
            
            log_p_fmri_flat = log_p_fmri.view(-1, V)  # [B*T, V]
            p_img_flat = p_img.view(-1, V)            # [B*T, V]
            
            # Select only valid tokens
            log_p_fmri_masked = log_p_fmri_flat[mask_flat]  # [N_valid, V]
            p_img_masked = p_img_flat[mask_flat]            # [N_valid, V]
            
            # Compute KL divergence on valid tokens only
            loss = self.kl_loss(log_p_fmri_masked, p_img_masked)
        else:
            # No mask - compute on all tokens
            loss = self.kl_loss(log_p_fmri, p_img)
        
        return loss



class EarlyAlignmentLoss(nn.Module):
    """
    Combined loss for Early Alignment Stage 2
    
    Stage 1: Caption loss only (LLM training)
    Stage 2: λ_align * Alignment + λ_cciea * CCIEA
    
    Alignment options:
    - MSE (Euclidean)
    - Hyperbolic Distance
    - Hyperbolic Contrastive
    """
    
    def __init__(
        self,
        # Stage 2 loss weights
        lambda_align: float = 1.0,
        lambda_cciea: float = 0.15,
        lambda_caption: float = 0.0,
        
        # Hyperparameters
        curvature: float = 0.1,
        cciea_temperature: float = 1.0,
        contrastive_temperature: float = 0.1,
        
        # Config
        stage_config: Optional[StageConfig] = None
    ):
        super().__init__()
        
        # Loss weights
        self.lambda_align = lambda_align
        self.lambda_cciea = lambda_cciea 
        self.lambda_caption = lambda_caption
        
        self.stage_config = stage_config or StageConfig.get_stage2_early_mse()
        
        # Initialize alignment loss based on configuration
        if self.stage_config.use_hyperbolic_infonce:
            self.align_loss = HyperbolicInfoNCELossVectorized(
                curvature=curvature,
                temperature=contrastive_temperature
            )
            print("  Using: Hyperbolic InfoNCE Loss (Proposed)")
        elif self.stage_config.use_contrastive:
            # Hyperbolic Contrastive Loss
            self.align_loss = HyperbolicContrastiveLoss(
                curvature=curvature,
                temperature=contrastive_temperature
            )
            print("  Using: Hyperbolic Contrastive Loss")
        elif self.stage_config.use_hyperbolic:
            self.align_loss = HyperbolicAlignmentLoss(curvature)
            print("  Using: Hyperbolic Distance Loss")
        else:
            # Euclidean alignment (ablation)
            self.align_loss = nn.MSELoss()
            print("  Using: MSE Alignment Loss")
        
        self.cciea_loss = CCIEALoss(temperature=cciea_temperature)
        
        # Store curvature for target projection 
        self.curvature = curvature
        self.hyp_ops = HyperbolicOperations(c=curvature) if self.stage_config.use_hyperbolic else None
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all losses with ablation support
        
        Args:
            outputs: model outputs
                - lat_image_pred: (B, 257, 1024) predicted CLIP features (Euclidean)
                - z_image_pred: (B, 257, 1024) predicted CLIP in hyperbolic space
                - cciea_logits_fmri: fMRI path caption logits
                - cciea_logits_img: image path caption logits (detached)
                - cciea_mask: caption mask
                - caption_loss: CE caption loss (optional)
            targets: ground truth tokens
                - lat_image: (B, 257, 1024) GT CLIP features (Euclidean)
                - z_image: (B, 257, 1024) GT CLIP in hyperbolic space
        
        Returns:
            losses: dictionary of all loss components
        """
        
        losses = {}
        total_loss = 0.0
        
        # ====================================================================
        # STAGE 1: Caption Loss Only
        # ====================================================================
        if self.stage_config.stage == 1:
            if 'caption_loss' in outputs:
                loss_caption = outputs['caption_loss']
                losses['caption'] = loss_caption
                total_loss += loss_caption
        
        # ====================================================================
        # STAGE 2: Alignment + CCIEA + Caption
        # ====================================================================
        elif self.stage_config.stage == 2:
        
            # 1. Alignment Loss (Main) - Compare in SAME space (Image space)
            if self.stage_config.use_hyperbolic_infonce:
                # Hyperbolic InfoNCE: hyperbolic space embeddings 
                if 'z_image_pred' in outputs and 'z_image' in targets and 'sample_idx' in targets:
                    loss_align = self.align_loss(
                        outputs['z_image_pred'],   # fMRI → hyperbolic
                        targets['z_image'],        # Image → hyperbolic
                        targets['sample_idx']
                    )
                    losses['alignment'] = loss_align
                    total_loss += self.lambda_align * loss_align
            
            elif self.stage_config.use_contrastive:
                # Hyperbolic Contrastive Loss
                if 'z_image_pred' in outputs and 'z_image' in targets:
                    loss_align = self.align_loss(outputs['z_image_pred'], targets['z_image'], targets['sample_idx']) # [B, 257, 1024] fMRI predicted, [B, 257, 1024] GT CLIP features in hyp space
                    losses['alignment'] = loss_align
                    total_loss += self.lambda_align * loss_align
            
            elif self.stage_config.use_hyperbolic:
                # Hyperbolic Distance Loss
                if 'z_image_pred' in outputs and 'z_image' in targets:
                    loss_align = self.align_loss(outputs['z_image_pred'], targets['z_image'])        # [B, 257, 1024]
                    losses['alignment'] = loss_align
                    total_loss += self.lambda_align * loss_align
            else:
                # MSE Loss (Euclidean)
                if 'lat_image_pred' in outputs and 'lat_image' in targets:
                    loss_align = self.align_loss(outputs['lat_image_pred'],  targets['lat_image'])   # [B, 257, 1024]
                    losses['alignment'] = loss_align
                    total_loss += self.lambda_align * loss_align
            
            # 2. CCIEA Loss (Caption-Conditioned Image Embedding Alignment)
            if self.lambda_cciea > 0:
                if 'cciea_logits_fmri' in outputs and 'cciea_logits_img' in outputs:
                    loss_cciea = self.cciea_loss(
                        logits_fmri=outputs['cciea_logits_fmri'],
                        logits_img=outputs['cciea_logits_img'].detach(),  # Teacher is detached!
                        caption_mask=outputs.get('cciea_mask', None)
                    )
                    losses['cciea'] = loss_cciea
                    total_loss += self.lambda_cciea * loss_cciea
            
            # 3. Caption Loss (Optional, for end-to-end supervision)
            if self.lambda_caption > 0:
                if 'caption_loss' in outputs:
                    loss_caption = outputs['caption_loss']
                    losses['caption'] = loss_caption
                    total_loss += self.lambda_caption * loss_caption
        
        losses['total'] = total_loss
        return losses
