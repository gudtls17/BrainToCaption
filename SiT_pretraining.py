"""
SiT Self-Supervised Pretraining for fMRI Embeddings (Updated for Tokenized Input)
==================================================================================

Masked Autoencoding approach for pretraining SiT encoder
Updated to use the modified SiT model with (batch, token, dim) input
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict
from einops import repeat
from os.path import join
import gc

# Import modified SiT
from sit import SiT


# ============================================================================
# 1. MASKED AUTOENCODER WITH SiT (UPDATED)
# ============================================================================

class SiTMaskedAutoencoder(nn.Module):
    """
    Masked Autoencoder using SiT encoder and decoder
    
    Architecture:
    1. Mask random tokens from fMRI input
    2. Encode visible tokens with SiT encoder
    3. Add mask tokens for masked positions
    4. Decode all tokens with SiT decoder
    5. Reconstruct masked tokens
    
    Uses modified SiT with tokenized input: (batch, num_tokens, token_dim)
    """
    
    def __init__(
        self,
        # Data parameters (for tokenized input)
        input_token_dim: int = 64,  # Dimension of input tokens
        num_tokens: int = 1000,         # Number of tokens (patches)
        
        # Encoder parameters
        encoder_dim: int = 384,
        encoder_depth: int = 12,
        encoder_heads: int = 6,
        encoder_dim_head: int = 64,
        encoder_mlp_ratio: int = 2,
        
        # Decoder parameters
        decoder_dim: int = 384,
        decoder_depth: int = 6,
        decoder_heads: int = 6,
        decoder_dim_head: int = 64,
        decoder_mlp_ratio: int = 2,
        
        # Training parameters
        mask_ratio: float = 0.75,
        use_pe: str = 'sin-cos',
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
    ):
        super().__init__()
        
        self.num_tokens = num_tokens
        self.mask_ratio = mask_ratio
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.input_token_dim = input_token_dim
        
        # ====================================================================
        # ENCODER (Using modified SiT)
        # ====================================================================
        
        self.encoder = SiT(
            dim=encoder_dim,
            depth=encoder_depth,
            heads=encoder_heads,
            num_tokens=num_tokens,
            token_dim=input_token_dim,  # Input token dimension
            num_classes=encoder_dim,    # Dummy, will use get_embedding()
            dim_head=encoder_dim_head,
            dropout=dropout,
            emb_dropout=emb_dropout,
            mlp_ratio=encoder_mlp_ratio,
            use_pe=use_pe,
            use_class_token=True,
            weights_layers_init=False,
            trainable_pos_emb=False,
            no_class_token_emb=True
        )
        
        # ====================================================================
        # DECODER (Using modified SiT)
        # ====================================================================
        
        self.decoder = SiT(
            dim=decoder_dim,
            depth=decoder_depth,
            heads=decoder_heads,
            num_tokens=num_tokens,
            token_dim=decoder_dim,      # Decoder works in its own space
            num_classes=input_token_dim,  # Reconstruct original tokens
            dim_head=decoder_dim_head,
            dropout=dropout,
            emb_dropout=emb_dropout,
            mlp_ratio=decoder_mlp_ratio,
            use_pe=use_pe,
            use_class_token=True,
            weights_layers_init=False,
            trainable_pos_emb=False,
            no_class_token_emb=True
        )
        
        # ====================================================================
        # ENCODER-DECODER BRIDGE
        # ====================================================================
        
        # Project encoder output to decoder dimension
        self.encoder_to_decoder = nn.Linear(encoder_dim, decoder_dim, bias=True)
        
        # Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        nn.init.normal_(self.mask_token, std=0.02)
    
    def random_masking(
        self,
        x: torch.Tensor,
        mask_ratio: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Random masking of tokens
        
        Args:
            x: [B, N, D] - input tokens
            mask_ratio: ratio of tokens to mask
        
        Returns:
            x_masked: [B, N_visible, D] - visible tokens
            mask: [B, N] - binary mask (0: keep, 1: remove)
            ids_restore: [B, N] - indices to restore original order
        """
        
        B, N, D = x.shape
        num_keep = int(N * (1 - mask_ratio))
        
        # Random noise for shuffling
        noise = torch.rand(B, N, device=x.device)
        
        # Sort noise to get random permutation
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep first num_keep tokens
        ids_keep = ids_shuffle[:, :num_keep]
        
        # Gather visible tokens
        x_masked = torch.gather(
            x, 
            dim=1, 
            index=ids_keep.unsqueeze(-1).expand(-1, -1, D)
        )
        
        # Binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, N], device=x.device)
        mask[:, :num_keep] = 0
        
        # Unshuffle to get original mask order
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def forward_encoder(
        self,
        x: torch.Tensor,
        mask_ratio: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder with masking
        
        Args:
            x: [B, N, D] - fMRI input tokens
            mask_ratio: masking ratio (uses self.mask_ratio if None)
        
        Returns:
            latent: [B, encoder_dim] - encoded representation (CLS token)
            visible_tokens: [B, N_visible, encoder_dim] - encoded visible tokens
            mask: [B, N] - binary mask
            ids_restore: [B, N] - restore indices
        """
        
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        
        B, N, D = x.shape
        
        # Random masking (before feeding to encoder)
        x_visible, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        # Get encoder token embeddings (without transformer yet)
        # We need to manually process to get token-level outputs
        x_embed = self.encoder.to_patch_embedding(x_visible)  # [B, N_visible, encoder_dim]
        
        # Add positional embeddings (only for visible tokens)
        n_visible = x_visible.shape[1]
        
        # Get positional embeddings for visible positions
        # First, get the original positions of visible tokens
        ids_keep = torch.argsort(ids_restore, dim=1)[:, :n_visible]
        
        if self.encoder.pos_embedding is not None:
            # Select positional embeddings for visible tokens
            pos_embed_visible = torch.gather(
                self.encoder.pos_embedding.expand(B, -1, -1),
                dim=1,
                index=ids_keep.unsqueeze(-1).expand(-1, -1, self.encoder_dim)
            )
            x_embed = x_embed + pos_embed_visible
        
        # Add CLS token
        cls_token = repeat(self.encoder.cls_token, '1 1 d -> b 1 d', b=B)
        x_embed = torch.cat([cls_token, x_embed], dim=1)
        
        # Apply dropout
        x_embed = self.encoder.dropout(x_embed)
        
        # Apply transformer
        x_transformed = self.encoder.transformer(x_embed)
        
        # Extract CLS token and visible tokens
        latent = x_transformed[:, 0, :]  # [B, encoder_dim]
        visible_tokens = x_transformed[:, 1:, :]  # [B, N_visible, encoder_dim]
        
        return latent, visible_tokens, mask, ids_restore
    
    def forward_decoder(
        self,
        visible_tokens: torch.Tensor,
        ids_restore: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through decoder
        
        Args:
            visible_tokens: [B, N_visible, encoder_dim] - encoded visible tokens
            ids_restore: [B, N] - indices to restore token order
        
        Returns:
            pred: [B, N, token_dim] - reconstructed tokens
        """
        
        B = visible_tokens.shape[0]
        N = ids_restore.shape[1]
        
        # Project encoder output to decoder dimension
        x = self.encoder_to_decoder(visible_tokens)  # [B, N_visible, decoder_dim]
        
        # Append mask tokens
        n_visible = x.shape[1]
        n_mask = N - n_visible
        
        mask_tokens = repeat(
            self.mask_token, 
            '1 1 d -> b n d',
            b=B,
            n=n_mask
        )
        
        # Concatenate visible + mask tokens
        x = torch.cat([x, mask_tokens], dim=1)  # [B, N, decoder_dim]
        
        # Unshuffle to restore original order
        x_unshuffle = torch.gather(
            x,
            dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, self.decoder_dim)
        )
        
        # Now we have tokens in original order, feed to decoder
        # Decoder will add its own positional embeddings and CLS token
        
        # Manually add decoder positional embeddings
        if self.decoder.pos_embedding is not None:
            x_unshuffle = x_unshuffle + self.decoder.pos_embedding[:, :N, :]
        
        # Add decoder CLS token
        decoder_cls_token = repeat(self.decoder.cls_token, '1 1 d -> b 1 d', b=B)
        x_unshuffle = torch.cat([decoder_cls_token, x_unshuffle], dim=1)
        
        # Apply dropout
        x_unshuffle = self.decoder.dropout(x_unshuffle)
        
        # Apply decoder transformer
        x_decoded = self.decoder.transformer(x_unshuffle)
        
        # Remove CLS token
        x_decoded = x_decoded[:, 1:, :]  # [B, N, decoder_dim]
        
        # Predict tokens
        pred = self.decoder.mlp_head(x_decoded)  # [B, N, token_dim]
        
        return pred
    
    def forward_loss(
        self,
        target: torch.Tensor,
        pred: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute reconstruction loss (MSE) on masked tokens only
        
        Args:
            target: [B, N, D] - original fMRI tokens
            pred: [B, N, D] - predicted tokens
            mask: [B, N] - binary mask (1: compute loss, 0: ignore)
        
        Returns:
            loss: scalar loss
            loss_per_token: [B, N] - loss per token (for analysis)
        """
        
        # MSE loss per token
        loss_per_token = (pred - target) ** 2
        loss_per_token = loss_per_token.mean(dim=-1)  # [B, N]
        
        # Compute loss only on masked tokens
        loss = (loss_per_token * mask).sum() / mask.sum()
        
        return loss, loss_per_token
    
    def forward(
        self,
        x: torch.Tensor,
        mask_ratio: Optional[float] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass: encode, decode, compute loss
        
        Args:
            x: [B, N, D] - fMRI input tokens
            mask_ratio: masking ratio
        
        Returns:
            Dictionary containing loss and intermediate outputs
        """
        
        # Encode with masking
        latent, visible_tokens, mask, ids_restore = self.forward_encoder(x, mask_ratio)
        
        # Decode
        pred = self.forward_decoder(visible_tokens, ids_restore)
        
        # Compute loss
        loss, loss_per_token = self.forward_loss(x, pred, mask)
        
        return {
            'loss': loss,
            'pred': pred,
            'mask': mask,
            'latent': latent,
            'visible_tokens': visible_tokens,
            'loss_per_token': loss_per_token
        }
    
    def get_encoder_embedding(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract encoder embeddings without masking (for downstream tasks)
        
        Args:
            x: [B, N, D] - fMRI input tokens
        
        Returns:
            embeddings: [B, encoder_dim] - fMRI embeddings
        """
        
        with torch.no_grad():
            # Use SiT's get_embedding method (no masking)
            embedding = self.encoder.get_embedding(x)
        
        return embedding


# ============================================================================
# 2. TRAINING UTILITIES
# ============================================================================

class SiTPretrainingDataset(torch.utils.data.Dataset):
    """Dataset for SiT pretraining with tokenized fMRI input"""
    
    def __init__(self, data_path: str, normalize: bool = False):
        """
        Args:
            data_path: Path to the .npy file containing [num_tokens, token_dim] tokenized fMRI data
            normalize: whether to z-normalize (Not use. already normalized)
        """
        
        self.fmri_paths = data_path         # list of file paths to .npy files
        self.num_samples = len(data_path)
        
    #     if normalize:
    #         self._normalize()
    
    # def _normalize(self):
    #     """Z-normalize per token dimension across samples"""
    #     mean = self.fmri_tokens.mean(dim=0, keepdim=True)
    #     std = self.fmri_tokens.std(dim=0, keepdim=True) + 1e-8
    #     self.fmri_tokens = (self.fmri_tokens - mean) / std
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        fmri_data_mmap = np.load(self.fmri_paths[idx], mmap_mode='r')  
        fmri_data = fmri_data_mmap.copy()
        fmri_data = torch.from_numpy(fmri_data).float()
        
        return fmri_data


def train_sit_pretraining(
    model: SiTMaskedAutoencoder,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    num_epochs: int = 100,
    learning_rate: float = 1e-4,
    warmup_epochs: int = 10,
    device: str = 'cuda',
    save_path: str = 'V:/XXXXX/Project/Language_decoding/1.sit_pretraining/',
    log_interval: int = 10
):
    """
    Train SiT with masked autoencoding
    """
    
    model = model.to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.05
    )
    
    # Learning rate scheduler with warmup
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            # Cosine decay after warmup
            progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
            return 0.5 * (1.0 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training loop
    best_val_loss = float('inf')
    train_history = []
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_losses = []
        
        for batch_idx, x in enumerate(train_loader):
            x = x.to(device)
            
            # Forward
            outputs = model(x)
            loss = outputs['loss']
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
            
            if batch_idx % log_interval == 0:
                print(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f}")
        
        avg_train_loss = np.mean(train_losses)
        
        # Validation
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for x in val_loader:
                x = x.to(device)
                outputs = model(x)
                val_losses.append(outputs['loss'].item())
        
        avg_val_loss = np.mean(val_losses)
        
        # Update scheduler
        scheduler.step()
        
        # Log
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        train_history.append({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'lr': optimizer.param_groups[0]['lr']
        })
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
            }, join(save_path, 'sit_pretrained.pth'))
            print(f"  -> Saved best model (val_loss: {avg_val_loss:.4f})")
        
        print()
    
    print(f"Pretraining complete! Best val loss: {best_val_loss:.4f}")
    
    return train_history


# ============================================================================
# 3. EXTRACT PRETRAINED EMBEDDINGS
# ============================================================================

def extract_sit_embeddings(
    pretrained_model: SiTMaskedAutoencoder,
    fmri_tokens_path, 
    batch_size: int = 32,
    device: str = 'cuda'
) -> np.ndarray:
    """
    Extract embeddings using pretrained SiT encoder
    
    Args:
        pretrained_model: Pretrained SiTMaskedAutoencoder
        fmri_tokens_path: fMRI data path
        batch_size: batch size for extraction
        device: device to use
    
    Returns:
        embeddings: [N, encoder_dim] numpy array
    """
    
    pretrained_model = pretrained_model.to(device)
    pretrained_model.eval()
    
    dataset = SiTPretrainingDataset(fmri_tokens_path, normalize=False)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    all_embeddings = []
    
    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            embeddings = pretrained_model.get_encoder_embedding(x)
            all_embeddings.append(embeddings.cpu().numpy())
    
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    
    print(f"Extracted embeddings: {all_embeddings.shape}")
    
    return all_embeddings


# ============================================================================
# 4. EXAMPLE USAGE
# ============================================================================

def example_sit_pretraining():
    """Example of SiT pretraining workflow with tokenized input"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_path = '/store9/NSD/nsddata_betas/ppdata/subj01/fsaverage'
    save_path='/store10/XXXXX/Project/Language_decoding/1.sit_pretraining'
    
    # # 1. Create dummy data (replace with actual fMRI tokens)
    # print("Creating dummy tokenized fMRI data...")
    # n_train = 5000
    # n_val = 1000
    # num_tokens = 320  # Number of tokens per sample
    # token_dim = 153  # Dimension of each token
    
    # # Shape: (batch, num_tokens, token_dim)
    # train_data = np.random.randn(n_train, num_tokens, token_dim).astype(np.float32)
    # val_data = np.random.randn(n_val, num_tokens, token_dim).astype(np.float32)
    
    # 1. Load your tokenized fMRI data here
    print("Loading tokenized fMRI filepath...")
    
    # Get NSD ID
    train_nsdid_fmri=pd.read_pickle(join('/store9/NSD', 'nsddata_betas/ppdata/subj01/fsaverage', 'subj1_train_nsd-img_to_fmri.pkl')) # key: nsd_img_id, value: fmri trial index
    test_nsdid_fmri=pd.read_pickle(join('/store9/NSD', 'nsddata_betas/ppdata/subj01/fsaverage', 'subj1_test_nsd-img_to_fmri.pkl'))   # using key(NSD ID) only

    train_nsdid_fmri=np.array(list(train_nsdid_fmri.keys()))
    test_nsdid_fmri=np.array(list(test_nsdid_fmri.keys()))
    
    path = '/store9/NSD/nsddata_betas/ppdata/subj01/fsaverage/3.1.fMRI_Scha1000_normalized_average_concat/'
    ids_str_train = np.char.zfill(train_nsdid_fmri.astype('str'), 5)
    result_train = np.char.add(ids_str_train, '_LR_train.npy')
    result_train = np.char.add('fmri_average_nsdID_', result_train)
    # rights_train = np.char.add(ids_str_train, '_R_train.npy')
    # rights_train = np.char.add('fmri_average_nsdID_', rights_train)
    # result_train = np.column_stack((lefts_train, rights_train)).ravel()
    fmri_train_paths = np.char.add(path, result_train)
    
    ids_str_val = np.char.zfill(test_nsdid_fmri.astype('str'), 5)
    result_val = np.char.add(ids_str_val, '_LR_test.npy')
    result_val = np.char.add('fmri_average_nsdID_', result_val)
    # rights_val = np.char.add(ids_str_val, '_R_test.npy')
    # rights_val = np.char.add('fmri_average_nsdID_', rights_val)
    # result_val = np.column_stack((lefts_val, rights_val)).ravel()
    fmri_val_paths = np.char.add(path, result_val)
    
    data_sample = np.load(fmri_train_paths[0])      # Shape: (num_tokens, token_dim)
    num_tokens = data_sample.shape[0]
    token_dim = data_sample.shape[1]
    
    print(f"Train data number : {len(fmri_train_paths)}")
    print(f"Val data number : {len(fmri_val_paths)}")
    
    # 2. Create datasets and loaders
    train_dataset = SiTPretrainingDataset(fmri_train_paths, normalize=False)
    val_dataset = SiTPretrainingDataset(fmri_val_paths, normalize=False)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=64,
        num_workers=4
    )
    
    # 3. Create model
    print("\nInitializing SiT Masked Autoencoder...")
    model = SiTMaskedAutoencoder(
        input_token_dim=token_dim,                      # 64
        num_tokens=num_tokens,                          # 1000
        encoder_dim=384,
        encoder_depth=12,
        encoder_heads=6,
        encoder_dim_head=64,
        encoder_mlp_ratio=2,
        decoder_dim=384,
        decoder_depth=6,
        decoder_heads=6,
        decoder_dim_head=64,
        decoder_mlp_ratio=2,
        mask_ratio=0.75,
        use_pe='sin-cos'
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # 4. Train
    print("\nStarting pretraining...")
    history = train_sit_pretraining(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=100,                   # 100
        learning_rate=1e-4,
        warmup_epochs=10,
        device=device,
        save_path=save_path
    )
    
    # 5. Load best model and extract embeddings
    print("\nExtracting embeddings from pretrained model...")
    checkpoint = torch.load(join(save_path, 'sit_pretrained.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Extract embeddings for validation data
    embeddings = extract_sit_embeddings(
        pretrained_model=model,
        fmri_tokens_path=fmri_val_paths,
        batch_size=32,
        device=device
    )
    
    print(f"\nPretraining complete!")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embeddings saved to sit_embeddings.npy")
    
    np.save(join(save_path, 'sit_embeddings.npy'), embeddings)
    
    return model, history, embeddings


if __name__ == "__main__":
    model, history, embeddings = example_sit_pretraining()