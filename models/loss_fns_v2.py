import torch
import torch.nn as nn
import torch.nn.functional as F

class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.loss_fn = nn.MultiLabelSoftMarginLoss()

    def forward(self, input, target):
        return self.loss_fn(input, target)


class WeightedBCELoss(nn.Module):
    """
    Weighted BCE Loss for Multi-Label Classification
    
    对稀有类给予更高的权重，帮助模型更好地学习稀有类
    """
    def __init__(self, class_freq=None, gamma=1.0):
        super(WeightedBCELoss, self).__init__()
        self.gamma = gamma  # 控制权重的强度
        self.class_weights = None
        
        if class_freq is not None:
            self._compute_weights(class_freq)
    
    def _compute_weights(self, class_freq):
        """
        计算类别权重
        使用 inverse frequency: w_c = (N / n_c)^gamma
        """
        class_freq = torch.tensor(class_freq, dtype=torch.float32)
        total = class_freq.sum()
        # 避免除零
        class_freq = torch.clamp(class_freq, min=1)
        # Inverse frequency weighting
        weights = (total / class_freq) ** self.gamma
        # 归一化
        weights = weights / weights.mean()
        self.register_buffer('class_weights', weights)
    
    def forward(self, input, target):
        """
        input: (bs, nc) logits
        target: (bs, nc) multi-hot
        """
        # 使用 sigmoid + BCE
        pred = torch.sigmoid(input)
        
        # BCE loss per element
        loss = F.binary_cross_entropy(pred, target.float(), reduction='none')
        
        # 应用类别权重
        if self.class_weights is not None:
            loss = loss * self.class_weights.unsqueeze(0)
        
        return loss.mean()


class FocalBCELoss(nn.Module):
    """
    Focal Loss for Multi-Label Classification
    
    对难分类的样本给予更高的权重
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalBCELoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, input, target):
        pred = torch.sigmoid(input)
        
        # Focal weight
        pt = target * pred + (1 - target) * (1 - pred)
        focal_weight = (1 - pt) ** self.gamma
        
        # Alpha weighting
        alpha_t = target * self.alpha + (1 - target) * (1 - self.alpha)
        
        # BCE loss
        bce = F.binary_cross_entropy(pred, target.float(), reduction='none')
        
        loss = alpha_t * focal_weight * bce
        return loss.mean()


class ASLLoss(nn.Module):
    """
    Asymmetric Loss for Multi-Label Classification
    
    对正样本和负样本使用不同的 gamma
    正样本少，需要更关注；负样本多，可以适当忽略简单负样本
    """
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05):
        super(ASLLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
    
    def forward(self, input, target):
        pred = torch.sigmoid(input)
        
        # Asymmetric Clipping
        pred_pos = pred
        pred_neg = (pred - self.clip).clamp(min=0)
        
        # 正负样本分别处理
        loss_pos = target * torch.log(pred_pos.clamp(min=1e-8))
        loss_neg = (1 - target) * torch.log((1 - pred_neg).clamp(min=1e-8))
        
        # Asymmetric Focusing
        pt_pos = pred_pos
        pt_neg = 1 - pred_neg
        
        weight_pos = (1 - pt_pos) ** self.gamma_pos
        weight_neg = (1 - pt_neg) ** self.gamma_neg
        
        loss = -(weight_pos * loss_pos + weight_neg * loss_neg)
        return loss.mean()
