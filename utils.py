import torch
from typing import Optional

def construct_mask(source, target, source_pad_idx, target_pad_idx, device: Optional[torch.device] = None):
    
    source_pad_mask = (source != source_pad_idx).unsqueeze(1).unsqueeze(2)
    target_pad_mask = (target != target_pad_idx).unsqueeze(1).unsqueeze(2)

    target_length = target.shape[1]
    target_sequence_mask = torch.tril(torch.ones((target_length, target_length))).bool()
    target_sequence_mask = target_sequence_mask.to(device)
    target_mask = target_sequence_mask & target_pad_mask

    
    return source_pad_mask, target_mask