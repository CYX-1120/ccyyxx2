import torch
import torch.nn as nn
import torch.nn.functional as F

class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.loss_fn = nn.MultiLabelSoftMarginLoss()

    def forward(self, input, target):
        # input: bs, nc; the output of model without sigmoid
        # target: bs, nc; multi-hot format

        return self.loss_fn(input, target)


class AdaptiveWeightedBCELoss(nn.Module):
    """
    Learnable weighted loss for mixed samples.
    
    Core idea:
    - Original samples: weight = 1.0 (reliable labels)
    - Mixed samples: weight learned based on label complexity
    - More labels in mixed sample → less reliable → lower weight
    """
    def __init__(self, num_classes=20):
        super(AdaptiveWeightedBCELoss, self).__init__()
        self.num_classes = num_classes
        
        # Learnable parameters for mix weight
        # weight = sigmoid(base - decay * label_count)
        self.mix_weight_base = nn.Parameter(torch.tensor(2.0))    # Initial high value
        self.mix_weight_decay = nn.Parameter(torch.tensor(0.15))  # Decay per label
        
        # Minimum weight to prevent collapse
        self.min_weight = 0.3
        
    def forward(self, input, target, mix_ind=None):
        """
        Args:
            input: [B, C] logits
            target: [B, C] multi-hot labels
            mix_ind: [B] tensor, 0 for original, 1 for mixed samples
        """
        # Per-sample BCE loss (reduction='none')
        loss_per_sample = F.multilabel_soft_margin_loss(
            input, target, reduction='none'
        ).mean(dim=1)  # [B]
        
        if mix_ind is None:
            # No mix info, use standard loss
            return loss_per_sample.mean()
        
        # Separate original and mixed samples
        original_mask = (mix_ind == 0)
        mixed_mask = (mix_ind == 1)
        
        # Original samples: weight = 1.0
        loss_original = loss_per_sample[original_mask]
        
        # Mixed samples: adaptive weight based on label count
        if mixed_mask.sum() > 0:
            mixed_targets = target[mixed_mask]
            label_counts = (mixed_targets > 0).float().sum(dim=1)  # [num_mixed]
            
            # Compute weights: sigmoid(base - decay * label_count)
            raw_weights = torch.sigmoid(
                self.mix_weight_base - self.mix_weight_decay * label_counts
            )
            # Clamp to minimum
            weights = torch.clamp(raw_weights, min=self.min_weight)
            
            loss_mixed = loss_per_sample[mixed_mask]
            weighted_loss_mixed = (weights * loss_mixed).sum() / (weights.sum() + 1e-6)
        else:
            weighted_loss_mixed = torch.tensor(0.0, device=input.device)
        
        # Combine losses
        n_orig = original_mask.sum().float()
        n_mix = mixed_mask.sum().float()
        
        if n_orig > 0 and n_mix > 0:
            # Weighted average
            total_loss = (loss_original.mean() * n_orig + weighted_loss_mixed * n_mix) / (n_orig + n_mix)
        elif n_orig > 0:
            total_loss = loss_original.mean()
        else:
            total_loss = weighted_loss_mixed
        
        # Regularization: prevent weights from collapsing
        if mixed_mask.sum() > 0:
            weight_reg = F.relu(self.min_weight - raw_weights.mean()) * 0.1
            total_loss = total_loss + weight_reg
        
        return total_loss
    
    def get_current_weights(self):
        """For logging purposes"""
        return {
            'base': self.mix_weight_base.item(),
            'decay': self.mix_weight_decay.item()
        }
