"""
LLaMA-3 Caption Generation Module
==================================

Features:
1. LLaMA-3-8B for caption generation from fMRI latent embeddings
2. Training with teacher forcing (no autoregressive loop)
3. Inference with autoregressive generation
4. Multiple ground-truth captions support (5 captions per fMRI)
5. Standard caption metrics (CIDEr, BLEU, METEOR, SPICE)
"""

import os
from os.path import join
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # only use when execute this file directly

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    GenerationConfig,
)
from typing import Dict, List, Optional, Tuple
import numpy as np
from tqdm import tqdm
from huggingface_hub import login

# Optional: Caption evaluation metrics
try:
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    print("Warning: pycocoevalcap not installed. Metrics will not be available.")

SAVE_PATH = '/store10/XXXXX/Project/Language_decoding/2.result/dummy_result'


# ============================================================================
# LLAMA-3 CAPTION GENERATOR
# ============================================================================

class LLaMA3CaptionGenerator(nn.Module):
    """
    LLaMA-3-8B based caption generator
    
    Input: fMRI latent embedding (B, llm_dim) from DualHyperbolicBrainCaptionModel
    Output: Caption text
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B",
        llm_input_dim: int = 4096,
        max_caption_length: int = 50,  # Based on training set statistics
        use_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        device: str = 'cuda'
    ):
        super().__init__()
        
        self.llm_input_dim = llm_input_dim
        self.max_caption_length = max_caption_length
        self.device = device
        
        # ====================================================================
        # Load LLaMA-3 Tokenizer
        # ====================================================================
        print(f"Loading LLaMA-3 tokenizer from {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            padding_side='right'  # Important for causal LM
        )
        
        # Add pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Special tokens
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        
        print(f"  BOS token: {self.bos_token_id}")
        print(f"  EOS token: {self.eos_token_id}")
        print(f"  PAD token: {self.pad_token_id}")
        print(f"  Note: PAD=EOS, but EOS will be learned via label masking")
        
        # ====================================================================
        # Load LLaMA-3 Model
        # ====================================================================
        print(f"Loading LLaMA-3 model from {model_name}...")
        self.llama = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,  # Use BF16 for memory efficiency
            device_map=device,
            low_cpu_mem_usage=True,
            load_in_4bit=False,         # Quantization for memory efficiency
            # bnb_4bit_use_double_quant=True,
            # bnb_4bit_quant_type="nf4",
            # bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        self.llama.gradient_checkpointing_enable()  # Activate Gradient Checkpointing (Reduce VRAM usage)
        
        # Get hidden dimension
        self.llama_hidden_dim = self.llama.config.hidden_size  # 4096 for LLaMA-3-8B
        print(f"  LLaMA hidden dim: {self.llama_hidden_dim}")
        
        # ====================================================================
        # Projection Layer: fMRI latent -> LLaMA hidden space
        # ====================================================================
        self.latent_to_llama = nn.Sequential(
            nn.Linear(llm_input_dim, self.llama_hidden_dim * 2),
            nn.LayerNorm(self.llama_hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.llama_hidden_dim * 2, self.llama_hidden_dim),
            nn.LayerNorm(self.llama_hidden_dim)
        )
        
        # ====================================================================
        # Optional: LoRA for parameter-efficient fine-tuning
        # ====================================================================
        if use_lora:
            print("Applying LoRA to LLaMA-3...")
            try:
                from peft import get_peft_model, LoraConfig, TaskType
                
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=0.1,
                    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],   # Target lalyer of LoRA (LLaMA attention modules)
                    bias="none"
                )
                
                self.llama = get_peft_model(self.llama, lora_config)
                self.llama.print_trainable_parameters()
                self.using_lora = True  # Flag for LoRA usage
                
            except ImportError as e:
                print(f"❌ PEFT Import Error Details: {e}")  # Show detailed error
                print("Warning: peft not installed. LoRA will not be used.")
                print("   Falling back to FULL fine-tuning (requires more memory)")
                print("   Install with: pip install peft")
                self.using_lora = False  # Flag for LoRA usage
        else:
            print("LoRA disabled. Full fine-tuning mode.")
            self.using_lora = False
        
        # ====================================================================
        # Generation Config
        # ====================================================================
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_caption_length,
            min_new_tokens=3,
            num_beams=5,
            do_sample=False,                 # Greedy decoding for deterministic results
            temperature=1.0,                # Not use when do_sample=False
            top_p=1.0,                      # Not use when do_sample=False
            top_k=50,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,         # Prevent 3-gram repeat
            length_penalty=1.0,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            early_stopping=True
        )
    
    def forward(
        self,
        fmri_latent: torch.Tensor,
        caption_tokens: Optional[torch.Tensor] = None,
        caption_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            fmri_latent: (B, llm_input_dim) - fMRI latent from hyperbolic model
            caption_tokens: (B, T) - tokenized captions (for training)
            caption_mask: (B, T) - attention mask (for training)
        
        Returns:
            Dictionary containing logits and loss
        """
        
        B = fmri_latent.shape[0]
        
        # Project fMRI latent to LLaMA hidden space
        # (B, llm_input_dim) -> (B, llama_hidden_dim)
        fmri_embedding = self.latent_to_llama(fmri_latent)  # (B, L, 4096)
        
        # Change fMRI embedding dtype to LLaMA model dtype(bfloat16)
        fmri_embedding = fmri_embedding.to(dtype=self.llama.dtype, device=self.llama.device)
        
        # ====================================================================
        # Training Mode: Teacher Forcing
        # ====================================================================
        if caption_tokens is not None:
            # Add BOS embedding
            bos_ids = torch.full((B, 1), self.bos_token_id, dtype=torch.long, device=fmri_latent.device)
            bos_embeds = self.llama.get_input_embeddings()(bos_ids)
            
            # # Get token embeddings from LLaMA
            # # (B, T) -> (B, T, llama_hidden_dim)
            # instruction_embeds = self.llama.get_input_embeddings()(instruction_tokens)
            caption_embeds = self.llama.get_input_embeddings()(caption_tokens)
            
            # Concatenate fMRI embedding as prefix
            # [(B, K, 4096), # (B, T_inst, 4096), (B, T_cap, 4096)] -> (B, K+T_inst+T_cap, 4096)
            # inputs_embeds = torch.cat([fmri_embedding, instruction_embeds, caption_embeds], dim=1)
            inputs_embeds = torch.cat([bos_embeds, fmri_embedding, caption_embeds], dim=1)
            
            # Create attention mask
            if caption_mask is not None:
                # Prepend 1 for fMRI embedding
                bos_mask = torch.ones(B, 1, dtype=caption_mask.dtype, device=caption_mask.device)
                fmri_mask = torch.ones(B, fmri_embedding.shape[1], dtype=caption_mask.dtype, device=caption_mask.device)
                # inst_mask = torch.ones(B, instruction_tokens.shape[1], dtype=caption_mask.dtype, device=caption_mask.device)
                # attention_mask = torch.cat([fmri_mask, inst_mask, caption_mask], dim=1)  # (B, K+T_inst+T_cap)
                attention_mask = torch.cat([bos_mask, fmri_mask, caption_mask], dim=1)  # (B, K+T_inst+T_cap)
            else:
                attention_mask = None
            
            # Forward through LLaMA
            outputs = self.llama(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True
            )
            
            # Loss (next-token prediction)
            # Get logits: (B, 1+T, vocab_size)
            logits = outputs.logits
            
            # Shift for next-token prediction
            # Predict tokens [1:T] from inputs [0:T-1]
            # prefix_len = fmri_embedding.shape[1] + instruction_tokens.shape[1]  # fMRI + instruction
            prefix_len = 1+ fmri_embedding.shape[1]  # BOS + fMRI
            shift_logits = logits[:, prefix_len-1 :-1, :].contiguous()  # (B, T, vocab_size)
            
            # Learn only first EOS using Label masking
            shift_labels = caption_tokens.clone()  # (B, T)
            
            # token masking (-100) after first EOS
            for b in range(B):
                eos_positions = (caption_tokens[b] == self.eos_token_id).nonzero(as_tuple=True)[0]
                if len(eos_positions) > 0:
                    first_eos_idx = eos_positions[0].item()
                    # Remain first EOS (Can train), After EOS set to -100 (ignore)
                    if first_eos_idx + 1 < shift_labels.shape[1]:
                        shift_labels[b, first_eos_idx + 1:] = -100
                        
            shift_labels = shift_labels.contiguous()      # (B, T)
            
            # Compute loss (only on valid tokens)
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1, reduction='mean')
            
            # Flatten for loss computation
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),  # (B*T, vocab_size)
                shift_labels.view(-1)                           # (B*T,)
            )
            
            return {'loss': loss, 'logits': logits}
        
        # ====================================================================
        # Inference Mode: Autoregressive Generation
        # ====================================================================
        else:
            # ✅ Inference Mode (should NOT be used directly!)
            # Use generate_captions() instead
            raise NotImplementedError("Use generate_captions() for inference!")
    
    # ========================================================================
    # CCIEA Forward Method 
    # For Caption-Conditioned Image Embedding Alignment
    # ========================================================================
    def forward_for_cciea(
        self,
        latent_embedding: torch.Tensor,
        caption_tokens: torch.Tensor,
        caption_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for CCIEA loss computation.
        Returns caption logits without computing CE loss.
        
        This method is used for both:
        1. Image path (teacher): Get P(y | z_img)
        2. fMRI path (student): Get P(y | z_fmri)
        
        Args:
            latent_embedding: (B, K, D) - llm_input from model
                              (either from image or fMRI path)
            caption_tokens: (B, T) - tokenized GT captions
            caption_mask: (B, T) - attention mask
        
        Returns:
            Dictionary containing:
                - 'caption_logits': (B, T, V) - logits for caption tokens
                - 'caption_mask': (B, T) - valid token mask for CCIEA
        """
        
        B = latent_embedding.shape[0]
        
        # Project latent to LLaMA hidden space
        # (B, K, D) -> (B, K, 4096)
        projected_embedding = self.latent_to_llama(latent_embedding)
        projected_embedding = projected_embedding.to(
            dtype=self.llama.dtype, 
            device=self.llama.device
        )
        
        # Add BOS embedding
        bos_ids = torch.full((B, 1), self.bos_token_id, dtype=torch.long, device=latent_embedding.device)
        bos_embeds = self.llama.get_input_embeddings()(bos_ids)
        
        # Get caption token embeddings
        caption_embeds = self.llama.get_input_embeddings()(caption_tokens)
        
        # Concatenate: [BOS, latent_embedding, caption_embeds]
        # (B, 1+K+T, 4096)
        inputs_embeds = torch.cat([bos_embeds, projected_embedding, caption_embeds], dim=1)
        
        # Create attention mask
        if caption_mask is not None:
            bos_mask = torch.ones(B, 1, dtype=caption_mask.dtype, device=caption_mask.device)
            latent_mask = torch.ones(B, projected_embedding.shape[1], dtype=caption_mask.dtype, device=caption_mask.device)
            attention_mask = torch.cat([bos_mask, latent_mask, caption_mask], dim=1)
        else:
            attention_mask = None
        
        # Forward through LLaMA (frozen for CCIEA)
        outputs = self.llama(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Get logits: (B, 1+K+T, vocab_size)
        logits = outputs.logits
        
        # Extract caption logits only
        # prefix_len = BOS(1) + latent_tokens(K)
        prefix_len = 1 + projected_embedding.shape[1]
        
        # Caption logits: predict tokens [1:T] from positions [prefix:prefix+T-1]
        # shift_logits: (B, T, vocab_size)
        caption_logits = logits[:, prefix_len-1:-1, :].contiguous()
        
        # Create valid token mask (exclude padding after first EOS)
        valid_mask = caption_tokens.clone()
        for b in range(B):
            eos_positions = (caption_tokens[b] == self.eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                first_eos_idx = eos_positions[0].item()
                if first_eos_idx + 1 < valid_mask.shape[1]:
                    valid_mask[b, first_eos_idx + 1:] = 0
            # Convert to binary mask (1 for valid, 0 for pad)
            valid_mask[b] = (valid_mask[b] != self.pad_token_id).long()
            # Make sure positions up to first EOS are valid
            if len(eos_positions) > 0:
                valid_mask[b, :first_eos_idx + 1] = 1
        
        return {
            'caption_logits': caption_logits,  # (B, T, V)
            'caption_mask': valid_mask         # (B, T)
        }
    
    
    def generate_captions(self, fmri_latent, return_tokens=False):
        """
        Generate captions for the given latent embeddings (Inference Mode)
        
        Args:
            latent_embeds: (Batch, K, Dim) - Projected fMRI or Image embeddings
            return_tokens: whether to return token IDs
        
        Returns:
            List[str]: Generated captions
        """
        
        self.eval()
        
        B = fmri_latent.shape[0]
        
        fmri_embedding = self.latent_to_llama(fmri_latent)
        fmri_embedding = fmri_embedding.to(dtype=self.llama.dtype, device=self.llama.device)
        
        bos_ids = torch.full((B, 1), self.bos_token_id, dtype=torch.long, device=fmri_latent.device)
        bos_embeds = self.llama.get_input_embeddings()(bos_ids)
        
        inputs_embeds = torch.cat([bos_embeds, fmri_embedding], dim=1)
        
        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device)
        
        # 5. Generate
        with torch.no_grad():
            generated_ids = self.llama.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                generation_config=self.generation_config
            )
        
        # 6. Decode
        captions = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        
        if return_tokens:
            # Calculate Length (remove BOS)
            actual_lengths = []
            for ids in generated_ids:
                # Remove BOS because it is first token
                # fine EOS
                try:
                    # BOS to EOS
                    eos_pos = (ids[1:] == self.eos_token_id).nonzero(as_tuple=True)[0][0].item()
                    actual_lengths.append(eos_pos + 1)  # +1 for BOS
                except:
                    actual_lengths.append(len(ids))
            
            return captions, generated_ids, actual_lengths
        
        return captions
        


# ============================================================================
# CAPTION DATASET
# ============================================================================

class CaptionDataset(torch.utils.data.Dataset):
    """
    Dataset for LLaMA-3 caption training
    
    Input:
    - fMRI latent embeddings: (N, llm_dim)
    - Ground-truth captions: (N, K) where K=5 captions per fMRI
    """
    
    def __init__(
        self,
        fmri_latents: torch.Tensor,     # (N, llm_dim)
        caption_texts: List[List[str]],  # List of N samples, each with K captions
        tokenizer: AutoTokenizer,
        max_length: int = 32
    ):
        self.fmri_latents = fmri_latents
        self.caption_texts = caption_texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        self.num_samples = len(fmri_latents)
        self.num_captions_per_sample = len(caption_texts[0])
    
    def __len__(self):
        # Each sample has K captions, so total = N * K
        return self.num_samples * self.num_captions_per_sample
    
    def __getitem__(self, idx):
        # Map flat index to (sample_idx, caption_idx)
        sample_idx = idx // self.num_captions_per_sample
        caption_idx = idx % self.num_captions_per_sample
        
        # Get fMRI latent
        fmri_latent = self.fmri_latents[sample_idx]
        
        # Get caption text
        caption_text = self.caption_texts[sample_idx][caption_idx]
        
                
        tokens = self.tokenizer(
            caption_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            add_special_tokens=True, # Add <bos> and <eos> tokens
            return_tensors='pt'
        )
        
        return {
            'fmri_latent': fmri_latent,
            'caption_tokens': tokens['input_ids'].squeeze(0),      # (T,)
            'caption_mask': tokens['attention_mask'].squeeze(0),   # (T,)
            'caption_text': caption_text  # For reference
        }


def collate_caption_batch(batch):
    """Collate function for caption dataset"""
    
    fmri_latents = torch.stack([item['fmri_latent'] for item in batch])
    caption_tokens = torch.stack([item['caption_tokens'] for item in batch])
    caption_masks = torch.stack([item['caption_mask'] for item in batch])
    caption_texts = [item['caption_text'] for item in batch]
    
    return {
        'fmri_latent': fmri_latents,
        'caption_tokens': caption_tokens,
        'caption_mask': caption_masks,
        'caption_texts': caption_texts
    }


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_caption_generator(
    model: LLaMA3CaptionGenerator,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    device: str = 'cuda',
    save_path = join(SAVE_PATH, 'checkpoints')
):
    """
    Train LLaMA-3 caption generator
    """
    
    if hasattr(model, 'using_lora') and not model.using_lora:
        print("\n⚠️ WARNING: Training WITHOUT LoRA (Full fine-tuning)")
        print("   This requires significant GPU memory (~40GB for LLaMA-3-8B)")
        print("   Consider enabling LoRA with use_lora=True\n")
    
    model = model.to(device)
    model.train()
    
    # Optimizer (only trainable parameters)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=0.01
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=learning_rate * 0.1
    )
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # ================================================================
        # Training
        # ================================================================
        model.train()
        train_losses = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch in pbar:
            fmri_latent = batch['fmri_latent'].to(device)
            caption_tokens = batch['caption_tokens'].to(device)
            caption_mask = batch['caption_mask'].to(device)
            
            # Forward pass
            outputs = model(
                fmri_latent=fmri_latent,
                caption_tokens=caption_tokens,
                caption_mask=caption_mask
            )
            
            loss = outputs['loss']
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
            pbar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = np.mean(train_losses)
        
        # ================================================================
        # Validation
        # ================================================================
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                fmri_latent = batch['fmri_latent'].to(device)
                caption_tokens = batch['caption_tokens'].to(device)
                caption_mask = batch['caption_mask'].to(device)
                
                outputs = model(
                    fmri_latent=fmri_latent,
                    caption_tokens=caption_tokens,
                    caption_mask=caption_mask
                )
                
                val_losses.append(outputs['loss'].item())
        
        avg_val_loss = np.mean(val_losses)
        
        # Update scheduler
        scheduler.step()
        
        # Print summary
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            
            # ====================== 시간 오래 걸려서 잠시 주석 달음 ======================
            # torch.save({
            #     'epoch': epoch,
            #     'model_state_dict': model.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            #     'val_loss': avg_val_loss,
            # }, f"{save_path}/llama3_caption_best.pth")
            
            
            
            print(f"  → Saved best model (val_loss: {avg_val_loss:.4f})")
        
        print()
    
    print(f"Training complete! Best val loss: {best_val_loss:.4f}")


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def compute_caption_metrics(
    generated_captions: List[str],
    ground_truth_captions: List[List[str]]  # Multiple GT captions per sample
) -> Dict[str, float]:
    """
    Compute standard caption metrics
    
    Args:
        generated_captions: List of N generated captions
        ground_truth_captions: List of N samples, each with K GT captions
    
    Returns:
        Dictionary of metric scores
    """
    
    if not METRICS_AVAILABLE:
        print("Warning: pycocoevalcap not available. Returning dummy metrics.")
        return {'CIDEr': 0.0, 'BLEU-4': 0.0}
    
    # Format for pycocoevalcap
    # gts: {image_id: [caption1, caption2, ...]}
    # res: {image_id: [generated_caption]}
    
    gts = {}
    res = {}
    
    for i, (gen_cap, gt_caps) in enumerate(zip(generated_captions, ground_truth_captions)):
        gts[i] = gt_caps
        res[i] = [gen_cap]
    
    # Compute metrics
    metrics = {}
    
    # CIDEr (primary metric for image captioning)
    cider = Cider()
    metrics['CIDEr'], _ = cider.compute_score(gts, res)
    
    # BLEU
    bleu = Bleu(4)
    bleu_scores, _ = bleu.compute_score(gts, res)
    for i, score in enumerate(bleu_scores, 1):
        metrics[f'BLEU-{i}'] = score
    
    # METEOR
    try:
        meteor = Meteor()
        metrics['METEOR'], _ = meteor.compute_score(gts, res)
    except Exception as e:
        print(f"Warning: METEOR computation failed: {e}")
    
    # ROUGE-L
    try:
        rouge = Rouge()
        metrics['ROUGE-L'], _ = rouge.compute_score(gts, res)
    except:
        print("Warning: ROUGE-L computation failed")
    
    return metrics


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_caption_training():
    """Example of training LLaMA-3 caption generator"""
    
    from os.path import join
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # ====================================================================
    # 1. Prepare Data
    # ====================================================================
    
    # Dummy fMRI latents (from DualHyperbolicBrainCaptionModel output)
    # In practice, you would get these from the hyperbolic model
    num_train = 100
    num_val = 20
    llm_dim = 2048
    
    train_fmri_latents = torch.randn(num_train, llm_dim)
    val_fmri_latents = torch.randn(num_val, llm_dim)
    
    # Dummy captions (5 captions per fMRI)
    train_captions = [
        [
            "A person standing in a field",
            "Someone standing outdoors in grass",
            "A human figure in an open field",
            "A person in a grassy area",
            "An individual standing in nature"
        ] for _ in range(num_train)
    ]
    
    val_captions = [
        [
            "A dog playing with a ball",
            "A canine playing with a toy ball",
            "A dog having fun with a ball",
            "A playful dog and a ball",
            "A pet dog playing fetch"
        ] for _ in range(num_val)
    ]
    
    # ====================================================================
    # 2. Initialize Model
    # ====================================================================
    
    print("Initializing LLaMA-3 Caption Generator...")
    model = LLaMA3CaptionGenerator(
        model_name="meta-llama/Meta-Llama-3-8B",
        llm_input_dim=llm_dim,
        max_caption_length=80,
        use_lora=True,
        device=device
    )
    
    # ====================================================================
    # 3. Create Datasets
    # ====================================================================
    
    train_dataset = CaptionDataset(
        fmri_latents=train_fmri_latents,
        caption_texts=train_captions,
        tokenizer=model.tokenizer,
        max_length=32
    )
    
    val_dataset = CaptionDataset(
        fmri_latents=val_fmri_latents,
        caption_texts=val_captions,
        tokenizer=model.tokenizer,
        max_length=32
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_caption_batch
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_caption_batch
    )
    
    # ====================================================================
    # 4. Train
    # ====================================================================
    
    train_caption_generator(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=3,
        learning_rate=1e-4,
        device=device,
        save_path=join(SAVE_PATH, 'checkpoints')
    )
    
    # ====================================================================
    # 5. Generate Captions (Inference)
    # ====================================================================
    
    print("\n" + "="*60)
    print("Testing Caption Generation")
    print("="*60)
    
    model.eval()
    test_latent = val_fmri_latents[:5].to(device)                   # Take first 5 samples
    
    generated_captions = model.generate_captions(test_latent)       # Generate captions
    
    for i, caption in enumerate(generated_captions):
        print(f"\nSample {i+1}:")
        print(f"  Generated: {caption}")
        print(f"  GT Captions:")
        for j, gt_cap in enumerate(val_captions[i], 1):             # 5 ground truth captions per sample. Generated caption must match one of these.
            print(f"    {j}. {gt_cap}")
    
    # ====================================================================
    # 6. Compute Metrics
    # ====================================================================
    
    if METRICS_AVAILABLE:
        print("\n" + "="*60)
        print("Computing Metrics")
        print("="*60)
        
        metrics = compute_caption_metrics(
            generated_captions=generated_captions,
            ground_truth_captions=val_captions[:5]
        )
        
        for metric_name, score in metrics.items():
            print(f"{metric_name}: {score:.4f}")
    
    return model


if __name__ == "__main__":
    login(token='XX_XXXXXXXXXXXXXXXXXXXXXXXXXX')
    model = example_caption_training()
