import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import re
try:
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    print("Warning: pycocoevalcap not installed. Metrics will not be available.")
    
    
    
# ============================================================================
# EVALUATION METRICS
# ============================================================================

def compute_caption_metrics(
    generated_captions: List[str],
    ground_truth_captions: List[Union[List[str], np.ndarray]]  # Multiple GT captions per sample
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
    
    instruction_text = "Write a short, factual caption describing the main objects and actions in the image."
    
    for i, (gen_cap, gt_caps) in enumerate(zip(generated_captions, ground_truth_captions)):
        if isinstance(gt_caps, np.ndarray):
            gt_caps = gt_caps.tolist()  # numpy array → Python list
        elif not isinstance(gt_caps, list):
            gt_caps = [gt_caps]
        
        # 1. Ground Truth preprocessing: remove newlines and carriage returns
        clean_gt_caps = [str(c).replace('\n', ' ').replace('\r', ' ').strip() for c in gt_caps]
        
        # 2. Generated Caption preprocessing
        gen_clean = str(gen_cap).replace('\n', ' ').replace('\r', ' ')
        gen_clean = gen_clean.replace(instruction_text, "")                         # Remove instruction text if present
        
        gen_clean = re.sub(r'^[A-Z]{3,}', '', gen_clean)                            # Formula: Delete three or more consecutive ({3,}) capital letters([A-Z]) from sentence start (^)
        gen_clean = gen_clean.replace("Question", "").replace("definitely", "")
        gen_clean = gen_clean.strip()
        
        if not gen_clean:
            gen_clean = "."                                                         # prevent empty caption
        
        
        gts[i] = clean_gt_caps
        res[i] = [gen_clean]
    
    
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


def print_training_summary(
    epoch: int, 
    train_loss: float, 
    val_loss: float, 
    metrics: Dict[str, float], 
    gen_captions: List[str] = None,
    gt_captions: List[List[str]] = None,
    stage: int = 1,
    train_loss_dict: Dict[str, float] = None,
    val_loss_dict: Dict[str, float] = None
) -> None:
    """
    Training summary printout
    
    Args:
        epoch: Current epoch number
        train_loss: Average training loss
        val_loss: Average validation loss
        metrics: Caption evaluation metrics (CIDEr, BLEU, etc.)
        gen_captions: List of generated captions
        gt_captions: List of ground truth captions
        stage: Training stage (1 or 2)
        train_loss_dict: Detailed training losses (alignment, cycle, caption)
        val_loss_dict: Detailed validation losses (alignment, cycle, caption)
    """
    
    print(f"\n{'='*60}")
    print(f"Epoch {epoch} Summary")
    print(f"{'='*60}")
    
    # Loss
    print(f"\n[Loss]")
    print(f"  Train: {train_loss:.4f}")
    print(f"  Val:   {val_loss:.4f}")
    print(f"  Gap:   {val_loss - train_loss:.4f}")
    
    # Stage 2: Detailed loss breakdown
    if stage == 2 and (train_loss_dict or val_loss_dict):
        print(f"\n[Loss Breakdown]")
        print(f"  {'Component':<20} {'Train':>10} {'Val':>10}")
        print(f"  {'-'*40}")
        
        # Alignment Loss
        train_align = train_loss_dict.get('alignment', 0) if train_loss_dict else 0
        val_align = val_loss_dict.get('alignment', 0) if val_loss_dict else 0
        print(f"  {'Alignment (Hyper)':<20} {train_align:>10.4f} {val_align:>10.4f}")
        
        # CCIEA Loss 
        train_cciea = train_loss_dict.get('cciea', 0) if train_loss_dict else 0
        val_cciea = val_loss_dict.get('cciea', 0) if val_loss_dict else 0
        if train_cciea > 0 or val_cciea > 0:
            print(f"  {'CCIEA':<20} {train_cciea:>10.4f} {val_cciea:>10.4f}")
        
        # Caption Loss (CE)
        train_caption = train_loss_dict.get('caption', 0) if train_loss_dict else 0
        val_caption = val_loss_dict.get('caption', 0) if val_loss_dict else 0
        if train_caption > 0 or val_caption > 0:
            print(f"  {'Caption':<20} {train_caption:>10.4f} {val_caption:>10.4f}")
    
    # Caption Length (optional)
    if gen_captions and gt_captions:
        gen_lengths = [len(cap.split()) for cap in gen_captions]
        gt_lengths = [len(caps[0].split()) if isinstance(caps, (list, np.ndarray)) else len(caps.split()) for caps in gt_captions]
        print(f"\n[Caption Length]")
        print(f"  Generated: {sum(gen_lengths)/len(gen_lengths):.1f} words (min:{min(gen_lengths)}, max:{max(gen_lengths)})")
        print(f"  GT:        {sum(gt_lengths)/len(gt_lengths):.1f} words")
        print(f"  Ratio:     {sum(gen_lengths)/sum(gt_lengths):.2f}x")
    
    # Metrics with targets
    targets = {'CIDEr': 0.6388, 'BLEU-1': 0.6153, 'BLEU-2': 0.4345, 'BLEU-3': 0.3021, 'BLEU-4': 0.2192, 
               'METEOR': 0.2071, 'ROUGE-L': 0.4645}
    print(f"\n[Metrics]")
    for name, value in metrics.items():
        if name in targets:
            ratio = value / targets[name] * 100
            status = '✅' if ratio >= 100 else '🔄'
            print(f"  {name:8s}: {value:.4f} ({ratio:5.1f}% of target) {status}")
        else:
            print(f"  {name:8s}: {value:.4f}")
    
    print(f"{'='*60}\n")