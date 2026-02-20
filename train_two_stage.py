"""
Stage 1: Image → Caption (Vision-Language Grounding)
  - Train: Image Perceiver + LLM Projection + LLaMA (LoRA)
  - Frozen: CLIP
  
Stage 2: fMRI → Image Embedding → Caption (Brain Alignment)
  - Train: fMRI Perceiver + fMRI Projection
  - Frozen: Image Perceiver + LLM Projection + LLaMA + SiT
"""

import os
from os.path import join
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Set to use only GPU 0, it must be before importing torch
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import gc
from einops import rearrange, repeat
from datetime import datetime
from huggingface_hub import login

from losses import EarlyAlignmentLoss, StageConfig

try:
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    print("Warning: pycocoevalcap not installed. Metrics will not be available.")

from llama_caption_module import LLaMA3CaptionGenerator
from caption_eval_metric import compute_caption_metrics, print_training_summary
from hyperbolic_brain_caption import (
    HyperbolicBrainCaptionModel,
    NSDCaptionDataset,
    collate_caption_batch
)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed = 42
set_seed(seed)  



# ============================================================================
# TRAINING LOOP WITH BEST EPOCH SAVING
# ============================================================================
def train_one_epoch_stage1(model, caption_model, dataloader, criterion, optimizer, caption_optimizer, device, epoch, accumulate_steps: int = 2,
                    return_embeddings: bool = False,
                    ):
    
    
    """
    Stage 1 Training: Image → Caption
    
    TRAIN: Image Perceiver + LLM Projection + LLaMA (LoRA)
    FROZEN: None (CLIP is external)
    """
    
    model.train()
    caption_model.train()
    
    total_loss = 0.0
    loss_dict = {}              # Track all losses
    epoch_loss_history = []     # Step-wise history
    
    optimizer.zero_grad()
    caption_optimizer.zero_grad()
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]")):
        
        # Move to device
        image_tokens = batch['image_tokens'].to(device)                                                             # (B, 257, 1024)
        caption_tokens = batch['caption_tokens'].to(device)                                                         # (B, T)
        caption_mask = batch['caption_mask'].to(device)                                                             # (B, T)
        
        # Forward: Image → Perceiver → LLM Input
        outputs = model(image_tokens=image_tokens, stage=1)
        
        # Caption generation
        llm_input = outputs['llm_input']  # (B, K, llm_dim)
        
        # loss optimization is done inside caption model
        caption_outputs = caption_model(
            fmri_latent=llm_input,
            caption_tokens=caption_tokens,
            caption_mask=caption_mask
        )
        
        loss = caption_outputs['loss'] / accumulate_steps
        
        # Backward pass
        loss.backward()                 # Accumulate gradients
        
        
        if (batch_idx + 1) % accumulate_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(caption_model.parameters(), max_norm=1.0)
            
            optimizer.step()
            caption_optimizer.step()
            
            optimizer.zero_grad()
            caption_optimizer.zero_grad()
        
        total_loss += caption_outputs['loss'].item()  # Original value
        if 'caption' not in loss_dict:
            loss_dict['caption'] = 0.0
        loss_dict['caption'] += caption_outputs['loss'].item()  # Original loss
        
        # Step-wise logging
        step_log = {
            'epoch': epoch,
            'step': batch_idx,
            'caption_loss': caption_outputs['loss'].item()
        }
        epoch_loss_history.append(step_log)
        
        # Log
        if batch_idx % 500 == 0:
            print(f"  [{batch_idx}/{len(dataloader)}] Caption Loss: {caption_outputs['loss'].item():.4f}")
    
    # Epoch end: remaining gradient processing
    if (batch_idx + 1) % accumulate_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
    
    
    avg_loss = total_loss / len(dataloader)
    for k in loss_dict:
        loss_dict[k] /= len(dataloader)
    
    return avg_loss, loss_dict, epoch_loss_history


def train_one_epoch_stage2(
    model, caption_model, dataloader, criterion, optimizer, device, epoch, accumulate_steps: int = 4,
    use_caption_loss: bool = True, use_cciea: bool = True, return_embeddings: bool = True,
    caption_optimizer = None, end_to_end: bool = False
):
    """
    Stage 2 Training: fMRI → Predicted CLIP → Frozen Pipeline → Caption
    
    TRAIN: fMRI Perceiver + fMRI Projection
    FROZEN: Image Perceiver + LLM Projection + LLaMA + SiT
    
    Added CCIEA (Caption-Conditioned Image Embedding Alignment) loss support
    Alignment target: CLIP features (257, 1024) 
    """
    
    model.train()
    caption_model.eval()
    
    total_loss = 0.0
    loss_dict = {}
    epoch_loss_history = []
    
    optimizer.zero_grad()
    if caption_optimizer is not None:
        caption_optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]")):
        
        fmri_data = batch['fmri_data'].to(device)                      # (B, 1000, 64/384)
        target_image_tokens = batch['target_image_tokens'].to(device)  # (B, 257, 1024)
        sample_idx = batch['sample_idx'].to(device)
        
        # Fine features (optional)
        fine_features = batch.get('fine_features', None)
        if fine_features is not None:
            fine_features = fine_features.to(device)                    # (B, fine_dim)
        
        # ================================================================
        # 1. Get target image embedding (from frozen Image Perceiver)
        # ================================================================
        with torch.no_grad():
            # Euclidean target
            target_lat_image = target_image_tokens  # (B, 257, 1024)
            
            # Hyperbolic target (project CLIP features to hyperbolic space)
            if model.use_hyperbolic:
                u_image = model.hyp_proj_align(target_image_tokens)
                target_z_image = model.hyp_ops.exp_map(u_image)     # (B, 257, 1024)
            else:
                target_z_image = target_image_tokens
            
            # Also get Image Perceiver output for CCIEA
            if use_cciea:
                target_outputs = model(image_tokens=target_image_tokens, stage=1)
            
        # ================================================================
        # 2. Forward fMRI → Predicted CLIP features
        # ================================================================
        outputs = model(fmri_data=fmri_data, fine_features=fine_features, stage=2)                                                        
        
        # ================================================================
        # 3. Caption Loss + CCIEA Loss computation
        # ================================================================
        if 'caption_tokens' in batch:
            caption_tokens = batch['caption_tokens'].to(device)
            caption_mask = batch['caption_mask'].to(device)
        
            # 3.1 fMRI path Caption Loss (CE) - Gradient flows to fMRI encoder + CCIEA (fMRI logits)
            caption_outputs = caption_model(
                fmri_latent=outputs['llm_input'],
                caption_tokens=caption_tokens,
                caption_mask=caption_mask
            )
            
            if use_caption_loss:
                outputs['caption_loss'] = caption_outputs['loss']
            
            # 3.2 CCIEA Loss - Caption-Conditioned Alignment
            # Image path (teacher) vs fMRI path (student)
            # Student: fMRI path logits (gradient flows here)
            if use_cciea:
                full_logits = caption_outputs['logits']  # (B, 1+K+T, V)
                prefix_len = 1 + outputs['llm_input'].shape[1]  # BOS + K
                outputs['cciea_logits_fmri'] = full_logits[:, prefix_len-1:-1, :].contiguous()
                
                # Teacher: Image path logits (frozen, no gradient)
                with torch.no_grad():
                    img_cciea_outputs = caption_model.forward_for_cciea(
                        latent_embedding=target_outputs['llm_input'],  # Image path
                        caption_tokens=caption_tokens,
                        caption_mask=caption_mask
                    )
                outputs['cciea_logits_img'] = img_cciea_outputs['caption_logits']
                outputs['cciea_mask'] = img_cciea_outputs['caption_mask']
                
            
        # ================================================================
        # 4. Compute losses
        # ================================================================
        targets = {
            'z_image': target_z_image,          # GT image in hyperbolic space
            'lat_image': target_lat_image,      # GT image in Euclidean space
            'sample_idx': sample_idx,
        }
        
        losses = criterion(outputs, targets)
        loss = losses['total'] / accumulate_steps  # Normalize loss for gradient accumulation
        
        # Backward pass
        loss.backward()                 # Accumulate gradients
        
        if (batch_idx + 1) % accumulate_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            if caption_optimizer is not None:
                torch.nn.utils.clip_grad_norm_(caption_model.parameters(), max_norm=1.0)
                caption_optimizer.step()
                caption_optimizer.zero_grad()
            
        # Accumulate losses & Saving embedding for interpretation
        total_loss += losses['total'].item()  # Original value
        for k, v in losses.items():
            if k not in loss_dict:
                loss_dict[k] = 0.0
            loss_dict[k] += v.item()
        
        # Save current Step Loss 
        step_log = {
            'epoch': epoch,
            'step': batch_idx,
            'total_loss': losses['total'].item(), # Origianl value when not applying accumulation
            'caption_loss': outputs.get('caption_loss', torch.tensor(0.0)).item(),
            'alignment_loss': losses.get('alignment', torch.tensor(0.0)).item(),
            'cciea_loss': losses.get('cciea', torch.tensor(0.0)).item()
        }
        epoch_loss_history.append(step_log)
        
        # Log
        if batch_idx % 500 == 0:
            print(f"  [{batch_idx}/{len(dataloader)}] Total: {loss.item():.4f} | "
                  f"Align: {losses.get('alignment', 0):.4f} | "
                  f"CCIEA: {losses.get('cciea', 0):.4f} | "
                  f"Caption: {outputs.get('caption_loss', 0):.4f}")
    
    # Epoch end: remaining gradient processing
    if (batch_idx + 1) % accumulate_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
    
    
    # Average losses
    avg_loss = total_loss / len(dataloader)
    for k in loss_dict:
        loss_dict[k] /= len(dataloader)
    
    
    return avg_loss, loss_dict, epoch_loss_history

# ============================================================================
# VALIDATION LOOP (Loss Calculation Only)
# ============================================================================
def validate_one_epoch(
    model, caption_model, dataloader, criterion, device, stage, epoch, 
    return_embeddings: bool = False, save_captions: bool = True
):
    """
    Validation for both stages with caption generation
    """
    model.eval()
    caption_model.eval()
    
    total_loss = 0.0
    loss_dict = {}
    
    generated_captions = []
    ground_truth_captions = []
    NSD_ids = []
    
    all_embeddings = {}                                                     
    if return_embeddings:
        keys_to_save = [
            'lat_image', 'z_image', 'lat_fmri', 'z_fmri', 
            'lat_image_pred', 'z_image_pred', 'lat_semantic',
            'llm_input', 'attn_fmri'
        ]
        all_embeddings = {k: [] for k in keys_to_save}
    
    with torch.no_grad(): # Deactivate Gradient  (Save memory and stop training)
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1} [Val]")):
            
            if stage == 1:
                # Stage 1 validation
                image_tokens = batch['image_tokens'].to(device)
                caption_tokens = batch['caption_tokens'].to(device)
                caption_mask = batch['caption_mask'].to(device)
                
                outputs = model(image_tokens=image_tokens, stage=1)
                
                caption_outputs = caption_model(
                    fmri_latent=outputs['llm_input'],
                    caption_tokens=caption_tokens,
                    caption_mask=caption_mask
                )
                
                loss = caption_outputs['loss']
                total_loss += loss.item()
                
                if 'caption' not in loss_dict:
                    loss_dict['caption'] = 0.0
                loss_dict['caption'] += loss.item()
                
                if save_captions:
                    gen_caps = caption_model.generate_captions(outputs['llm_input'])
                    generated_captions.extend(gen_caps)
                    
                    # Get GT captions (all 5)
                    if 'captions' in batch:
                        ground_truth_captions.extend(batch['captions'])
                        
                    if 'ID' in batch:
                        ids = batch['ID']
                        NSD_ids.extend(ids)
            
            elif stage == 2:
                # Stage 2 validation
                fmri_data = batch['fmri_data'].to(device)
                target_image_tokens = batch['target_image_tokens'].to(device)
                sample_idx = batch['sample_idx'].to(device)
                
                # Fine features (optional)
                fine_features = batch.get('fine_features', None)
                if fine_features is not None:
                    fine_features = fine_features.to(device)
                
                caption_tokens = batch['caption_tokens'].to(device)
                caption_mask = batch['caption_mask'].to(device)
                
                # Get target
                target_lat_image = target_image_tokens
                if model.use_hyperbolic:
                    u_image = model.hyp_proj_align(target_image_tokens)
                    target_z_image = model.hyp_ops.exp_map(u_image)
                else:
                    target_z_image = target_image_tokens
                
                target_outputs = model(image_tokens=target_image_tokens, stage=1)
                
                # Forward fMRI
                outputs = model(fmri_data=fmri_data, fine_features=fine_features, stage=2)    # predicted image embeddings (lat_image_pred)
                
                # CCIEA computation
                caption_outputs = caption_model(
                    fmri_latent=outputs['llm_input'],
                    caption_tokens=caption_tokens,
                    caption_mask=caption_mask
                )
                outputs['caption_loss'] = caption_outputs['loss']
                
                # CCIEA Loss computation
                # fMRI path logits
                full_logits_fmri = caption_outputs['logits']  # (B, 1+K+T, V)
                prefix_len = 1 + outputs['llm_input'].shape[1]  # BOS + K
                outputs['cciea_logits_fmri'] = full_logits_fmri[:, prefix_len-1:-1, :].contiguous()
                
                # Image path logits (Teacher)
                img_cciea_outputs = caption_model.forward_for_cciea(
                    latent_embedding=target_outputs['llm_input'],
                    caption_tokens=caption_tokens,
                    caption_mask=caption_mask
                )
                outputs['cciea_logits_img'] = img_cciea_outputs['caption_logits']
                outputs['cciea_mask'] = img_cciea_outputs['caption_mask']
                
                # Compute losses
                targets = {                             # Ground truth for losses
                    'z_image': target_z_image,
                    'lat_image': target_lat_image,
                    'sample_idx': sample_idx,
                }
                
                losses = criterion(outputs, targets)
                loss = losses['total']
                
                total_loss += loss.item()
                for k, v in losses.items():
                    if k not in loss_dict:
                        loss_dict[k] = 0.0
                    loss_dict[k] += v.item()
                    
                if save_captions:
                    gen_caps = caption_model.generate_captions(outputs['llm_input'])
                    generated_captions.extend(gen_caps)
                    
                    # Get GT captions (all 5)
                    if 'captions' in batch:
                        ground_truth_captions.extend(batch['captions'])
                        
                    if 'ID' in batch:
                        ids = batch['ID']
                        NSD_ids.extend(ids)
            
            if return_embeddings:
                for key in all_embeddings.keys():
                    if key in outputs:
                        all_embeddings[key].append(outputs[key].detach().cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    for k in loss_dict:
        loss_dict[k] /= len(dataloader)
    
    caption_data = {
        'generated': generated_captions,
        'ground_truth': ground_truth_captions,
        'NSD_ids': NSD_ids
    } if save_captions else None
    
    collected_embeddings  = all_embeddings if return_embeddings else None
    
    return avg_loss, loss_dict, caption_data, collected_embeddings


# ============================================================================
# MAIN TRAINING FUNCTION 
# ============================================================================
def verify_tokenization(tokenizer, caption, max_length=50):
    
    # Add EOS
    caption_with_eos = caption.strip() + tokenizer.eos_token
    
    tokens = tokenizer(
        caption_with_eos,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        add_special_tokens=False,
        return_tensors='pt'
    )
    
    token_ids = tokens['input_ids'][0]
    
    print(f"Caption: '{caption}'")
    print(f"With EOS: '{caption_with_eos}'")
    print(f"Token IDs (first 15): {token_ids[:15].tolist()}")
    print(f"EOS token ID: {tokenizer.eos_token_id}")
    
    # Chekc EOS location
    eos_positions = (token_ids == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
    if len(eos_positions) > 0:
        print(f"✅ EOS found at position {eos_positions[0].item()}")
        return True
    else:
        print("❌ EOS NOT found!")
        return False

# Example usage
# verify_tokenization(caption_model.tokenizer, "A herd of cows grazing in a field.")


def train_two_stage(
    data_path: str,
    save_path: str,
    stage: int = 1,
    num_epochs: int = 3,
    batch_size_train: int = 24,
    batch_size_val: int = 24,
    learning_rate: float = 1e-4,
    num_perceiver_tokens: int = 16,
    device: str = 'cuda',
    sit_mode: str = 'frozen',
    image_layer_location = 'LastLayer',                             # 'LastLayer', 'SecondLastLayer', 'CLS'
    stage1_checkpoint: Optional[str] = None,
    experiment_type: str = 'early_mse'  # Experiment configuration
):
    """
    Main training function 
    
    Args:
        data_path: Path to data
        save_path: Base save path
        stage: 1 (Image→Caption) or 2 (fMRI→Caption)
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to use
        sit_mode: SiT mode ( 'finetune', 'frozen', 'precomputed')
        stage1_checkpoint: Path to Stage 1 weights (required for Stage 2)
        experiment_type: Experiment configuration for Stage 2
            - 'early_mse': MSE + CCIEA at (257, 1024) level
            - 'early_hyperbolic': Hyperbolic Distance + CCIEA
            - 'early_contrastive': Hyperbolic Contrastive + CCIEA
    """
    
    # current time for unique save folder (ex: 20251222_195000)
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"stage{stage}_SiT_{sit_mode}_Perceiver{num_perceiver_tokens:02d}_{image_layer_location}_{current_time}"
    
    # Create save directory for this ablation
    stage_save_path  = join(save_path, run_name)
    os.makedirs(stage_save_path , exist_ok=True)
    caption_save_path = join(stage_save_path , 'generated_captions')
    
    print("\n" + "="*80)
    print(f"Training Stage {stage}")
    print(f"SiT Mode: {sit_mode}")
    print("="*80)
    
    # ====================================================================
    # Load Data
    # ====================================================================
    print("===== Loading DATA =====...")
    # Get NSD ID
    train_nsdid_fmri=pd.read_pickle(join('/store9/NSD', 'nsddata_betas/ppdata/subj01/fsaverage', 'subj1_train_nsd-img_to_fmri.pkl')) # key: nsd_img_id, value: fmri trial index
    test_nsdid_fmri=pd.read_pickle(join('/store9/NSD', 'nsddata_betas/ppdata/subj01/fsaverage', 'subj1_test_nsd-img_to_fmri.pkl'))   # using key(NSD ID) only
    train_nsdid_fmri=np.array(list(train_nsdid_fmri.keys()))
    test_nsdid_fmri=np.array(list(test_nsdid_fmri.keys()))
    ids_str_train = np.char.zfill(train_nsdid_fmri.astype('str'), 5)
    ids_str_val = np.char.zfill(test_nsdid_fmri.astype('str'), 5)
    
    
    print("Loading tokenized fMRI filepath...")
    fmri_filepath = '/store9/NSD/nsddata_betas/ppdata/subj01/fsaverage/3.1.fMRI_Scha1000_normalized_average_concat/'
    
    result_train = np.char.add(ids_str_train, '_LR_train.npy')
    result_train = np.char.add('fmri_average_nsdID_', result_train)
    fmri_train_paths = np.char.add(fmri_filepath, result_train)
    
    result_val = np.char.add(ids_str_val, '_LR_test.npy')
    result_val = np.char.add('fmri_average_nsdID_', result_val)
    fmri_val_paths = np.char.add(fmri_filepath, result_val)
    
    print(f"  fMRI: Train data number: {fmri_train_paths.shape}")
    print(f"  fMRI: Val data number: {fmri_val_paths.shape}")
    
    # === Load Fine Feature Data (optional) ===
    print("Loading fine feature filepath...")
    use_fine_features = True  # Set to False to disable fine features
    fine_dim = 12535          # fsaverage6 visual cortex vertices
    
    if use_fine_features:
        fine_filepath = '/store9/NSD/nsddata_betas/ppdata/subj01/fsaverage/2.1.fMRI_fineInfo_normalized_average_concat/'
        
        fine_train_result = np.char.add(ids_str_train, '_LR_train.npy')
        fine_train_result = np.char.add('fmri_average_fine_nsdID_', fine_train_result)
        fine_train_paths = np.char.add(fine_filepath, fine_train_result)
        
        fine_val_result = np.char.add(ids_str_val, '_LR_test.npy')
        fine_val_result = np.char.add('fmri_average_fine_nsdID_', fine_val_result)
        fine_val_paths = np.char.add(fine_filepath, fine_val_result)
        
        print(f"  FINE: Train data number: {fine_train_paths.shape}")
        print(f"  FINE: Val data number: {fine_val_paths.shape}")
    else:
        fine_train_paths = None
        fine_val_paths = None
        print("  FINE features: DISABLED")
    
    # === Load IMAGE Data ===
    print("Loading tokenized Image filepath...")
    print('Used layer : ', image_layer_location)          # 'LastLayer', 'SecondLastLayer', 'CLS' 
    image_filepath = f'/store9/NSD/nsddata_betas/ppdata/subj01/fsaverage/4.embed_file/image/{image_layer_location}/'
    
    add_image_fullname_train = np.char.add(ids_str_train, '_train.npy')
    add_image_fullname_train = np.char.add(f'image_{image_layer_location}_nsdID_', add_image_fullname_train)
    # result_image_fullname_train = np.column_stack((add_image_fullname_train, add_image_fullname_train)).ravel()
    image_train_paths = np.char.add(image_filepath, add_image_fullname_train)
    
    add_image_fullname_test = np.char.add(ids_str_val, '_test.npy')
    add_image_fullname_test = np.char.add(f'image_{image_layer_location}_nsdID_', add_image_fullname_test)
    # result_image_fullname_test = np.column_stack((add_image_fullname_test, add_image_fullname_test)).ravel()
    image_val_paths = np.char.add(image_filepath, add_image_fullname_test)
    
    print(f"  IMAGE embedding location: {image_layer_location}")
    print(f"  IMAGE: Train data number: {image_train_paths.shape}")
    print(f"  IMAGE: Val data number: {image_val_paths.shape}")
    
    # === Load CAPTION Data ===
    print("Loading caption filepath...")
    caption_filepath = f'/store9/NSD/nsddata_betas/ppdata/subj01/fsaverage/4.embed_file/text/'
    
    add_caption_fullname_train = np.char.add(ids_str_train, '_train.npy')
    add_caption_fullname_train = np.char.add(f'caption_nsdID_', add_caption_fullname_train)
    # result_caption_fullname_train = np.column_stack((add_caption_fullname_train, add_caption_fullname_train)).ravel()
    caption_train_paths = np.char.add(caption_filepath, add_caption_fullname_train)
    
    add_caption_fullname_test = np.char.add(ids_str_val, '_test.npy')
    add_caption_fullname_test = np.char.add(f'caption_nsdID_', add_caption_fullname_test)
    # result_caption_fullname_test = np.column_stack((add_caption_fullname_test, add_caption_fullname_test)).ravel()
    caption_val_paths = np.char.add(caption_filepath, add_caption_fullname_test)
    
    print(f"  CAPTION: Train data number: {caption_train_paths.shape}")
    print(f"  CAPTION: Val data number: {caption_val_paths.shape}")
        
        
    # ====================================================================
    # Initialize Model 
    # ====================================================================
    print("\nInitializing model...")
    model = HyperbolicBrainCaptionModel(
        image_token_dim=1024,
        fmri_raw_dim=64,
        fmri_token_dim=384,
        latent_dim=768,
        num_latents=num_perceiver_tokens,   # K for image perceiver
        fmri_num_latents=257,               # Match CLIP tokens
        use_hyperbolic=True,
        use_fine_features=use_fine_features,  # Enable fine features
        fine_dim=fine_dim,                    # 12535 for fsaverage6
        sit_mode=sit_mode,                          # finetune,   frozen,   disabled
        sit_pretrained_path=join('/store10/XXXXX/Project/Language_decoding/1.sit_pretraining/best_sit_pretrained.pth')
    ).to(device)
    
    # ====================================================================
    # Initialize Caption Model
    # ====================================================================
    print("Initializing caption model...")
    MAX_CAPTION_LENGTH = 50
    
    caption_model = LLaMA3CaptionGenerator(
        model_name="meta-llama/Meta-Llama-3-8B",
        llm_input_dim=4096,
        max_caption_length=MAX_CAPTION_LENGTH,
        use_lora=True,
        device=device
    ).to(device)
    
    verify_tokenization(caption_model.tokenizer, "A herd of cows grazing in a field.")
    
    print("Metric example computation test...")
    metric_sample = compute_caption_metrics(["A herd of cows grazing in a grassy field. There are mountains in the background."], 
                            [
            [
                "White cows eating grass under trees and the sky",
            "Many cows in a pasture with trees eating grass.",
            "A herd of cows graze on a field of sparse grass.",
            "a herd of white cows grazing on brush among the trees",
            "A herd of mostly white cows in a field with some trees."
            ]
        ])
    print("  Sample Metric:", metric_sample)
    # ====================================================================
    # Stage-Specific Setup
    # ====================================================================
    
    if stage == 1:
        print("\n🔥 Stage 1: Training Image Perceiver + LLaMA")
        
        # All trainable (CLIP is external)
        stage_config = StageConfig.get_stage1()
        
        # Optimizer for both models
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate
        )
        caption_optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, caption_model.parameters()),
            lr=5e-5,
            weight_decay=0.01
        )
        
        scheduler = None
        criterion = None  # Not needed for Stage 1
        
        lambda_align = 0.0
        lambda_cciea = 0.0

    elif stage == 2:
        print("\n🔥 Stage 2: Training fMRI Perceiver ONLY")
        print(f"📊 Experiment Type: {experiment_type}")
        
         # End-to-End without Stage 1
        if experiment_type == 'end_to_end':
            print("⚠️ ABLATION: End-to-End training WITHOUT Stage 1 pretraining")
            print("   Training ALL components from scratch")
            
            # Skip Stage 1 checkpoint loading
            # No freeze - train everything
            
            # Unfreeze ALL components
            print("🔥 Unfreezing ALL components (End-to-End)...")
            for param in model.parameters():
                param.requires_grad = True
            
            # ⭐ LLaMA: FROZEN (OOM)
            print("❄️ Freezing LLaMA to prevent OOM...")
            for param in caption_model.parameters():
                param.requires_grad = False
            
            # Print trainable params
            model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            caption_params = sum(p.numel() for p in caption_model.parameters() if p.requires_grad)
            print(f"   Trainable params (model): {model_params:,}")
            print(f"   Trainable params (caption): {caption_params:,} (should be 0)")
            
            # Need BOTH optimizers
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                eps=1e-10
            )
            caption_optimizer = None
            
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=num_epochs, eta_min=1e-6
            )
            
            # Loss config
            stage_config = StageConfig.get_stage2_early_hyperbolic()
            lambda_align = 1.0
            lambda_cciea = 0.0
            lambda_caption = 0.15
            
            end_to_end = True
            
        else:
            # Load Stage 1 weights
            if stage1_checkpoint and os.path.exists(stage1_checkpoint):
                print(f"Loading Stage 1 weights from: {stage1_checkpoint}")
                checkpoint = torch.load(stage1_checkpoint, map_location=device)
                state_dict = checkpoint['model_state_dict']
                state_dict = {k: v for k, v in state_dict.items() 
                            if not k.startswith('fmri_')
                            and not k.startswith('fine_encoder')
                            and not k.startswith('sit.')
                            and not k.startswith('hyp_proj_align')
                            and not k.startswith('reconstruction_decoder')}
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                
                print(f"  Missing keys: {len(missing_keys)}")
                print(f"  Unexpected keys: {len(unexpected_keys)}")
                caption_model.load_state_dict(checkpoint['caption_model_state_dict'], strict=False)
                
                del checkpoint
                gc.collect()
                torch.cuda.empty_cache() 
            else:
                print("⚠️ WARNING: No Stage 1 checkpoint loaded!")
            
            # Freeze everything except fMRI path
            print("❄️ Freezing Stage 1 components...")
            for param in model.image_encoder.parameters():
                param.requires_grad = False
            for param in model.hyp_proj_image.parameters():
                param.requires_grad = False
            for param in model.llm_proj.parameters():
                param.requires_grad = False
            for param in caption_model.parameters():
                param.requires_grad = False
            
            print("🔥 Unfreezing fMRI Perceiver + Projection + Decoder...")
            for param in model.fmri_encoder.parameters():
                param.requires_grad = True
            for param in model.hyp_proj_align.parameters():
                param.requires_grad = True
            for param in model.fmri_to_image_proj.parameters():         # fMRI→Image projection
                param.requires_grad = True
            if model.use_fine_features and model.fine_encoder is not None:
                print("🔥 Unfreezing Fine Feature Encoder...")
                for param in model.fine_encoder.parameters():
                    param.requires_grad = True
            
            
            model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            caption_params = sum(p.numel() for p in caption_model.parameters() if p.requires_grad)
            print(f"   Trainable params (model): {model_params:,}")
            print(f"   Trainable params (caption): {caption_params:,} (should be 0)")
            
            # Optimizer only for fMRI components
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=learning_rate,
                eps=1e-10
            )
            caption_optimizer = None
            
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=num_epochs, eta_min=1e-6
            )
            
            end_to_end = False
        
        # ====================================================================
        # 🆕 AUTO-CONFIGURE EXPERIMENT
        # ====================================================================
        if experiment_type == 'early_mse':
            print("  🆕 Config: MSE Alignment + CCIEA")
            stage_config = StageConfig.get_stage2_early_mse()
            lambda_align = 1.0
            lambda_cciea = 1.0
            lambda_caption = 0.0
        
        elif experiment_type == 'early_hyperbolic':
            print("  🆕 Config: Hyperbolic Distance Alignment + CCIEA")
            stage_config = StageConfig.get_stage2_early_hyperbolic()
            lambda_align = 1.0
            lambda_cciea = 1.0
            lambda_caption = 0.0
        
        elif experiment_type == 'w/o_cciea':
            print("  🆕 Config: Hyperbolic Distance Alignment + CE")
            stage_config = StageConfig.get_stage2_early_hyperbolic()
            lambda_align = 1.0
            lambda_cciea = 0.000001
            lambda_caption = 1.0
            
        elif experiment_type == 'early_mse_w/o_cciea':
            print("  🆕 Config: MSE Alignment + CE")
            stage_config = StageConfig.get_stage2_early_mse()
            lambda_align = 1.0
            lambda_cciea = 0.000001
            lambda_caption = 1.0
        
        elif experiment_type == 'w/o_image_prediction':
            print("  🆕 Config: Only CCIEA")
            stage_config = StageConfig.get_stage2_early_hyperbolic()
            lambda_align = 0.0
            lambda_cciea = 1.0
            lambda_caption = 0.0
        
        elif experiment_type == 'end_to_end':
            print("  🆕 Config: End-to-End (Defined above)")
        
        else:
            raise ValueError(f"Unknown experiment_type: {experiment_type}")
        
        print(f"  λ_align={lambda_align}, λ_cciea={lambda_cciea}, λ_caption={lambda_caption}")
        
        # Loss function
        criterion = EarlyAlignmentLoss(
            lambda_align=lambda_align,
            lambda_cciea=lambda_cciea,
            lambda_caption=lambda_caption,
            cciea_temperature=1.0,
            stage_config=stage_config
        )

    
    # ====================================================================
    # Create Dataset
    # ====================================================================
    print("\nCreating dataset...")
    train_dataset = NSDCaptionDataset(
        image_paths=image_train_paths,
        fmri_paths=fmri_train_paths,
        caption_paths=caption_train_paths,
        fine_paths=fine_train_paths,
        stage=stage,
        llama_tokenizer=caption_model.tokenizer,
        max_caption_length=MAX_CAPTION_LENGTH,
        use_all_captions=True
    )
        
    val_dataset = NSDCaptionDataset(
        image_paths=image_val_paths,
        fmri_paths=fmri_val_paths,
        caption_paths=caption_val_paths,
        fine_paths=fine_val_paths,
        stage=stage,
        llama_tokenizer=caption_model.tokenizer,
        max_caption_length=MAX_CAPTION_LENGTH,
        ID_list=ids_str_val,
        use_all_captions=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=4,              # Set CPU Threads
        pin_memory=True,            # If using GPU, pin memory for faster transfer (RAM -> GPU[VRAM])
        persistent_workers=False,    # Keep workers alive for the entire epoch
        worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id),
        generator=torch.Generator().manual_seed(seed),
        collate_fn=collate_caption_batch
    )
    
    val_loader = DataLoader(
        val_dataset,                                                                        
        batch_size=batch_size_val,
        shuffle=False, 
        num_workers=4,
        pin_memory=True,
        persistent_workers=False,    
        worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id),
        generator=torch.Generator().manual_seed(seed),
        collate_fn=collate_caption_batch
    )
    
    
    # ====================================================================
    # Training Loop with Best Epoch Tracking
    # ====================================================================
    best_train_avg_loss = float('inf')
    best_val_loss = float('inf')
    best_cider = float('-inf')
    best_epoch = -1
    # history = []
    patience = 10
    
    training_record = {
        "experiment_type": experiment_type,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "lambda_align": lambda_align,
            "lambda_cciea": lambda_cciea,
            "patience": patience,
            "batch_size": batch_size_train,
        },
        "history": []  
    }
    
    # variables to save best model states
    best_model_state = None
    best_caption_model_state = None
    best_metrics = None
    best_embeddings = None
    
    best_embeddings_path = join(stage_save_path, 'best_embeddings')
    os.makedirs(best_embeddings_path, exist_ok=True)
    
    epochs_without_improvement = 0
    for epoch in range(num_epochs):
        # Training
        print(f"\n[Epoch {epoch+1}/{num_epochs}] Training...")
        # Train
        if stage == 1:
            train_loss, train_loss_dict, train_history = train_one_epoch_stage1(
                model, caption_model, train_loader, criterion,
                optimizer, caption_optimizer, device, epoch, accumulate_steps=2,
            )
            # train_loss_dict = {'caption': train_loss}
        elif stage == 2:
            train_loss, train_loss_dict, train_history = train_one_epoch_stage2(
                model, caption_model, train_loader, criterion,
                optimizer, device, epoch, accumulate_steps=4,
                use_caption_loss=(criterion.lambda_caption > 0),  # ✅ Auto: True if lambda>0
                use_cciea=(criterion.lambda_cciea > 0),            # ✅ Auto: True if lambda>0
                caption_optimizer=caption_optimizer,
                end_to_end=end_to_end
            )
        
        print(f"\n[Epoch {epoch+1}] Training Summary:")
        print(f"  Average Loss: {train_loss:.4f}")
        for key, value in train_loss_dict.items():
            print(f"  {key}: {value:.4f}")
        
        # 2. Validation Step 
        print(f"[Epoch {epoch+1}/{num_epochs}] Validating...")
        val_loss, val_loss_dict, caption_data, val_embeddings = validate_one_epoch(
            model, caption_model, val_loader, criterion, device, stage, epoch,
            return_embeddings=True,
            save_captions=True
        )
        
        print(f"\n[Epoch {epoch+1}] Validation Summary:")
        print(f"  Average Loss: {val_loss:.4f}")
        for key, value in val_loss_dict.items():
            print(f"  {key}: {value:.4f}")
        
        # Evaluate (Compute Caption Metrics)
        metrics = {}
        if caption_data and len(caption_data['generated']) > 0:
            print(f"\n[Epoch {epoch+1}] Computing Caption Metrics...")
            metrics = compute_caption_metrics(
                caption_data['generated'],
                caption_data['ground_truth']
            )
            
            print_training_summary(
                epoch=epoch + 1,
                train_loss=train_loss,
                val_loss=val_loss,
                metrics=metrics,
                gen_captions=caption_data['generated'],
                gt_captions=caption_data['ground_truth'],
                stage=stage,
                train_loss_dict=train_loss_dict,
                val_loss_dict=val_loss_dict
            )
        
        
        # Save Caption Results (Every Epoch)
        if caption_data:
            caption_file = join(caption_save_path, f'captions_epoch_{epoch+1}.json')
            os.makedirs(caption_save_path, exist_ok=True)
            
            caption_results = []
            for i, (gen, gt) in enumerate(zip(caption_data['generated'], caption_data['ground_truth'])):
                # memmap/ndarray → Python list
                if isinstance(gt, np.ndarray):
                    gt_list = [str(cap) for cap in gt]  # numpy array → list of strings
                elif isinstance(gt, (list, tuple)):
                    gt_list = [str(cap) for cap in gt]  # list → list of strings
                else:
                    gt_list = [str(gt)]  # single value → list with one string
                
                nsd_id = None
                if 'NSD_ids' in caption_data:
                    nsd_ids_list = caption_data['NSD_ids']
                    if i < len(nsd_ids_list):
                        nsd_id = nsd_ids_list[i]
                
                caption_results.append({
                    'sample_id': i,
                    'NSD_id': nsd_id,
                    'generated': str(gen),
                    'ground_truth': gt_list
                })
            
            with open(caption_file, 'w', encoding='utf-8') as f:
                json.dump(caption_results, f, indent=4, ensure_ascii=False)
            print(f"\n✓ Saved captions to {caption_file}")
        
        
        # Track history
        if hasattr(model, 'fine_encoder') and model.fine_encoder is not None:
            fs = model.fine_encoder.fine_scale.item()
            print(f"[Fine Scale] {fs:.4f}")
        else:
            fs = 0.0
        
        epoch_result = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'fine_scale': fs,
            **{f'train_{k}': v for k, v in train_loss_dict.items()},  
            **{f'val_{k}': v for k, v in val_loss_dict.items()},      
            **metrics
        }
        training_record['history'].append(epoch_result)
        
        
        # Check if this is the best epoch
        if val_loss < best_val_loss:
            best_cider = metrics['CIDEr']
            best_train_loss = train_loss
            best_val_loss = val_loss
            best_epoch = epoch+1
            best_metrics = metrics
            
            # Save states
            # weight cloning to avoid GPU memory issues (Deep Copy) 
            # Delete previous best state (Prevent memory leakage)
            if best_model_state is not None:
                del best_model_state
                del best_caption_model_state
                gc.collect()
                torch.cuda.empty_cache()
            
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_caption_model_state = {k: v.cpu().clone() for k, v in caption_model.state_dict().items()}
            best_embeddings = val_embeddings
            
            # Save best captions
            if caption_data:
                best_caption_file = join(caption_save_path, f'BEST_captions_epoch_{epoch+1}.json')
                with open(best_caption_file, 'w', encoding='utf-8') as f:
                    json.dump(caption_results, f, indent=4, ensure_ascii=False)
            
            # Save step-wise training history for best epoch
            best_train_history_file = join(stage_save_path, f'BEST_train_steps_epoch_{epoch+1}.json')
            with open(best_train_history_file, 'w', encoding='utf-8') as f:
                json.dump(train_history, f, indent=4)
            
            print(f"\n✓ New best model! Epoch {epoch+1}, Val Loss: {val_loss:.4f}")
            
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1  # for early stopping (if implemented)
        
        # Memory cleanup
        # Remove validation data
        if caption_data is not None:
            del caption_data
        if val_embeddings is not None:
            del val_embeddings
        
        # Remove Training history
        if 'train_history' in locals():
            del train_history
            
        del caption_results
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Scheduler step
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            print(f"  📈 Learning Rate: {current_lr:.2e}")
        
        
        
        print(f"  💾 Memory cleaned (Epoch {epoch+1} complete)")
        print()
        
        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
        
    # ====================================================================
    # Post-Training: Save Best Model & Extract Embeddings
    # ====================================================================
    print(f"\n{'='*80}")
    print("Training Complete. Saving Best Model and Extracting Embeddings...")
    print(f"{'='*80}")
    if best_model_state is not None:
        # 1. Save Best Model to Disk (Only Once)
        save_file = join(stage_save_path, 'best_model.pth')
        print(f"Saving best model to {save_file}...")
        
        torch.save({
            'epoch': best_epoch,
            'model_state_dict': best_model_state,
            'caption_model_state_dict': best_caption_model_state,
            'train_loss': training_record['history'][best_epoch-1]['train_loss'],
            'val_loss': best_val_loss,
            'metrics': best_metrics,
            'stage': stage,
            'loss_weights':{
                'lambda_align': criterion.lambda_align if criterion else None,
                'lambda_cciea': criterion.lambda_cciea if criterion else None,
                'lambda_caption': criterion.lambda_caption if criterion else None,
            }
        }, save_file)
        
        print(f"\n✓ New best model saved (epoch {best_epoch}, loss: {best_val_loss:.4f})")
        
        # Save best embeddings
        if best_embeddings:
            for key, value_list in best_embeddings.items():
                if len(value_list) > 0:
                    concatenated = np.concatenate(value_list, axis=0)
                    np.save(join(best_embeddings_path, f'{key}_best.npy'), concatenated)
            print(f"✓ Saved best embeddings to {best_embeddings_path}")
    
    # Save full history
    history_file = join(stage_save_path, 'training_history.json')
    with open(history_file, 'w') as f:
        json.dump(training_record, f, indent=4)
    print(f"✓ Saved training history to {history_file}")

    print(f"\n{'='*80}")
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    if best_metrics:
        print("Best Metrics:")
        for k, v in best_metrics.items():
            print(f"  {k}: {v:.4f}")
    print(f"{'='*80}\n")
    
    # ====================================================================
    # Cleanup: GPU resources cleanup
    # ====================================================================
    print("Cleaning up GPU resources...")
    
    # DataLoader workers exit
    del train_loader, val_loader, train_dataset, val_dataset
    
    # remove variables
    if 'best_embeddings' in locals() and best_embeddings is not None:
        del best_embeddings
    if 'caption_data' in locals() and caption_data is not None:
        del caption_data
    
    gc.collect()
    
    # CUDA cache clean
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # 모든 CUDA 작업 완료 대기
    
    print("✓ GPU cleanup complete")
    
    
    return model, caption_model

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_run(stage=1, num_epochs=30, num_perceiver_tokens=16, experiment_type='early_mse', image_layer_location='LastLayer'):
    """
    Example: Run two-stage training
    
    Args:
        experiment_type: Experiment configuration for Stage 2
            - 'early_mse': MSE + CCIEA at CLIP level
            - 'early_hyperbolic': Hyperbolic Distance + CCIEA
            - 'early_contrastive': Hyperbolic Contrastive + CCIEA
    """
    
    print('='*80)
    print(f'EXPERIMENT: {experiment_type.upper()}')
    print('='*80)
    
    print('Total epochs:', num_epochs)
    print('Perceiver token K:', num_perceiver_tokens)
    
    from os.path import join
    
    MAIN_PATH = '/store10/'
    DATA_PATH = '/store9/'
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_path = join(DATA_PATH, 'NSD/NSD_registration/betas_registration/subj01')
    save_path = join(MAIN_PATH, 'XXXXX/Project/Language_decoding/2.result')
    
    print('Device:', device)
    print('Current cuda device:', torch.cuda.current_device())
    print('Count of using GPUs:', torch.cuda.device_count())
    
    SIT_MODE = 'frozen'     # 'finetune', 'frozen', 'precomputed', 'disabled'       For main
    # SIT_MODE = 'disabled'     # 'finetune', 'frozen', 'precomputed', 'disabled'     For ablation (w/o sit)
        
    model = None
    caption = None
    
    if stage==1:
        # Stage 1: Image → Caption
        print("="*80)
        print("STAGE 1: Vision-Language Grounding")
        print("="*80)
        
        model, caption = train_two_stage(
            data_path=data_path,
            save_path=save_path,
            stage=1,
            num_epochs=num_epochs,
            batch_size_train=32,
            batch_size_val=32,
            num_perceiver_tokens=num_perceiver_tokens,
            image_layer_location=image_layer_location,
            device=device
        )
    
    elif stage==2:
        # Stage 2: fMRI → Caption
        print("="*80)
        print("STAGE 2: Brain Alignment")
        print("="*80)
        
        # ===== Stage 1 path Main =====
        stage1_ckpt = join(save_path, 'stage1_SiT_frozen_Perceiver16_SecondLastLayer_20260123_202954/best_model.pth')     # For SecondLastLayer, K=16
        
        # ===== Stage 1 path Ablation =====
        # stage1_ckpt = join(save_path, 'stage1_SiT_frozen_Perceiver16_20260110_150114/best_model.pth')                     # For LastLayer
        # stage1_ckpt = join(save_path, 'stage1_SiT_frozen_Perceiver08_SecondLastLayer_20260125_093514/best_model.pth')     # For SecondLastLayer, K=8
        # stage1_ckpt = join(save_path, 'stage1_SiT_frozen_Perceiver32_SecondLastLayer_20260125_135239/best_model.pth')     # For SecondLastLayer, K=32
        
        model, caption = train_two_stage(
            data_path=data_path,
            save_path=save_path,
            stage=2,
            num_epochs=num_epochs,
            batch_size_train=16,
            batch_size_val=16,
            num_perceiver_tokens=num_perceiver_tokens,
            device=device,
            image_layer_location=image_layer_location,
            sit_mode=SIT_MODE,
            stage1_checkpoint=stage1_ckpt,
            experiment_type=experiment_type
        )
    
    
    # ====================================================================
    # Final Cleanup
    # ====================================================================
    print("\n" + "="*80)
    print("Final cleanup before exit...")
    print("="*80)
    
    # Remove model from GPU
    if model is not None:
        model.cpu()
        del model
    if caption is not None:
        if hasattr(caption, 'llama'):
            caption.llama.cpu()
        caption.cpu()
        del caption
    
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    print("✓ All resources released")
    
    return None, None



if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    
    login(token='XXXXX')            # Hugging Face login for LLaMA model access
    
    # ========================================================================
    # 🔬 EXPERIMENT SELECTION
    # ========================================================================
    # Choose experiment type:
    #   'early_mse'         - MSE + CCIEA at CLIP level
    #   'early_hyperbolic'  - Hyperbolic Distance + CCIEA
    #   'w/o_cciea'         - Hyperbolic Distance + CE (Caption loss)
    #   'hyperbolic_infonce'- Hyperbolic Contrastive using InfoNCE + CCIEA
    #   'end_to_end'        - w/o stage 1, full training in stage 2
    
    EXPERIMENT = 'early_hyperbolic'         # ✅ Change this to switch experiments
    IMAGE_LAYER = 'SecondLastLayer'         # 'LastLayer', 'SecondLastLayer', 'CLS'
    
    # Do not control EXPERIMENT (stage1 is not affected by EXPERIMENT)
    # example_run(stage=1, num_epochs=20, num_perceiver_tokens=16, experiment_type=EXPERIMENT, image_layer_location='SecondLastLayer')  # Stage1, K=16, Main
    # example_run(stage=2, num_epochs=20, num_perceiver_tokens=16, experiment_type=EXPERIMENT, image_layer_location='SecondLastLayer')  # Stage2, K=16, Main            
    
    # ===== Ablation Perceiver K =====
    # example_run(stage=1, num_epochs=20, num_perceiver_tokens=16, experiment_type=EXPERIMENT, image_layer_location='LastLayer')        # Stage1, image last layer 
    # example_run(stage=2, num_epochs=20, num_perceiver_tokens=16, experiment_type=EXPERIMENT, image_layer_location='LastLayer')        # # Stage2, image last layer 
    
    # example_run(stage=1, num_epochs=20, num_perceiver_tokens=8, experiment_type=EXPERIMENT, image_layer_location='SecondLastLayer')  # Stage1, K=8
    # example_run(stage=1, num_epochs=20, num_perceiver_tokens=32, experiment_type=EXPERIMENT, image_layer_location='SecondLastLayer')  # Stage1, K=32
    
    # example_run(stage=2, num_epochs=20, num_perceiver_tokens=8, experiment_type=EXPERIMENT, image_layer_location='SecondLastLayer')  # Stage2, K=8
    # example_run(stage=2, num_epochs=20, num_perceiver_tokens=32, experiment_type=EXPERIMENT, image_layer_location='SecondLastLayer')  # Stage2, K=32
    
    # ===== Ablation Loss =====
    # example_run(stage=2, num_epochs=20, num_perceiver_tokens=16, experiment_type='early_mse', image_layer_location='SecondLastLayer')  # Stage2, K=16, w/o hyperbolic
    # example_run(stage=2, num_epochs=20, num_perceiver_tokens=16, experiment_type='w/o_cciea', image_layer_location='SecondLastLayer')  # Stage2, K=16, w/o CCIEA
    # example_run(stage=2, num_epochs=20, num_perceiver_tokens=16, experiment_type='early_mse_w/o_cciea', image_layer_location='SecondLastLayer')  # Stage2, K=16, w/o hyperbolic + CCIEA
    
    # ===== Ablation fine encoder =====
    # Set use_fine_features=False
    # example_run(stage=2, num_epochs=20, num_perceiver_tokens=16, experiment_type='early_hyperbolic', image_layer_location='SecondLastLayer')  # Stage2, K=16, w/o fine encoder
    
    # ===== Ablation w/o stage 1 =====
    # example_run(stage=2, num_epochs=20, num_perceiver_tokens=16, experiment_type='end_to_end', image_layer_location='SecondLastLayer')  # Stage2, K=16, w/o stage 1, 
    
    # ===== Ablation w/o image prediction =====
    # example_run(stage=2, num_epochs=20, num_perceiver_tokens=16, experiment_type='w/o_image_prediction', image_layer_location='SecondLastLayer')  # Stage2, K=16, w/o image prediction
    
    # ===== Ablation w/o SiT =====
    # Set SIT_MODE='disabled'
    # example_run(stage=2, num_epochs=20, num_perceiver_tokens=16, experiment_type='early_hyperbolic', image_layer_location='SecondLastLayer')  # Stage2, K=16, w/o SiT
    