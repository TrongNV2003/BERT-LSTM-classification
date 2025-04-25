import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    """
    Cross Entropy Loss (CE Loss) cho bài toán single-label classification
    """
    def __init__(self, weight=None, reduction='mean'):
        super(CrossEntropyLoss, self).__init__()
        self.weight = weight
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Tensor shape (batch_size, num_classes)
            targets: Tensor shape (batch_size)
        """
        return F.cross_entropy(inputs, targets, weight=self.weight, reduction=self.reduction)


class WeightedCrossEntropyLoss(nn.Module):
    """
    Cross Entropy Loss với class weights cho bài toán single-label classification mất cân bằng nhãn
    """
    def __init__(self, class_weights, reduction='mean'):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.class_weights = torch.tensor(class_weights, dtype=torch.float)
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Tensor shape (batch_size, num_classes)
            targets: Tensor shape (batch_size)
        """
        return F.cross_entropy(inputs, targets, weight=self.class_weights, reduction=self.reduction)


class FocalLossForSingleLabel(nn.Module):
    """
    Focal Loss cho bài toán phân loại đơn nhãn
    Tập trung vào các mẫu khó phân loại hơn
    Công thức: -alpha * (1-pt)^gamma * log(pt)
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLossForSingleLabel, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Tensor shape (batch_size, num_classes)
            targets: Tensor shape (batch_size)
        """
        num_classes = inputs.size(1)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        pt = torch.exp(-ce_loss)
        
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            alpha_weight = torch.ones_like(focal_loss)
            if isinstance(self.alpha, (list, tuple)):
                assert len(self.alpha) == num_classes
                for i in range(num_classes):
                    alpha_weight[targets == i] = self.alpha[i]
            else:
                alpha_weight = torch.where(targets > 0, 1-self.alpha, self.alpha)
            focal_loss = alpha_weight * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class BinaryCrossEntropyLoss(nn.Module):
    """
    BCE Loss cho bài toán multi-label classification
    """
    def __init__(self, weight=None, reduction='mean'):
        super(BinaryCrossEntropyLoss, self).__init__()
        self.weight = weight
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Tensor shape (batch_size, num_classes)
            targets: Tensor shape (batch_size, num_classes)
        """
        return F.binary_cross_entropy_with_logits(
            inputs, targets, weight=self.weight, reduction=self.reduction
        )


class WeightedBinaryCrossEntropyLoss(nn.Module):
    """
    BCE Loss với class weights cho bài toán multi-label classification mất cân bằng nhãn
    """
    def __init__(self, class_weights, reduction='mean'):
        super(WeightedBinaryCrossEntropyLoss, self).__init__()
        self.class_weights = torch.tensor(class_weights, dtype=torch.float)
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Tensor shape (batch_size, num_classes)
            targets: Tensor shape (batch_size, num_classes)
        """
        # Tạo weight cho từng mẫu và class
        sample_weights = torch.ones_like(targets)
        for i in range(self.class_weights.size(0)):
            sample_weights[:, i] = self.class_weights[i]
            
        return F.binary_cross_entropy_with_logits(
            inputs, targets, weight=sample_weights, reduction=self.reduction
        )


class FocalLossForMultiLabel(nn.Module):
    """
    Focal Loss cho bài toán phân loại đa nhãn
    Tập trung vào các mẫu khó phân loại hơn
    Công thức: -alpha * (1-p_t)^gamma * log(p_t)
    Trong đó p_t = p nếu y=1, và p_t = 1-p nếu y=0
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLossForMultiLabel, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Tensor shape (batch_size, num_classes)
            targets: Tensor shape (batch_size, num_classes)
        """
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        probs = torch.sigmoid(inputs)
        
        p_t = torch.where(targets == 1, probs, 1-probs)
        alpha_weight = torch.where(targets == 1, self.alpha, 1-self.alpha)
        focal_weight = alpha_weight * (1 - p_t) ** self.gamma
        focal_loss = focal_weight * BCE_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class BCEWithPadMaskLoss(nn.Module):
    """
    BCE Loss với pad_mask để bỏ qua các giá trị padding turn trong dialogue
    """
    def __init__(self, reduction='none'):
        super(BCEWithPadMaskLoss, self).__init__()
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Tensor shape (batch_size, num_classes)
            labels: Tensor shape (batch_size, num_classes)
            pad_mask: Tensor shape (batch_size, num_classes)
        """
        loss_matrix = F.binary_cross_entropy_with_logits(logits, labels, reduction=self.reduction)
        loss = (loss_matrix * pad_mask).sum() / pad_mask.sum()
        return loss


class FocalLossWithPadMask(nn.Module):
    """
    Focal Loss với pad_mask để bỏ qua các giá trị padding turn trong dialogue
    """
    def __init__(self, gamma: float = 2, alpha: float = 0.25, reduction='none'):
        super(FocalLossWithPadMask, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Tensor shape (batch_size, num_classes)
            labels: Tensor shape (batch_size, num_classes)
            pad_mask: Tensor shape (batch_size, num_classes)
        """
        bce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction=self.reduction)
        probs = torch.sigmoid(logits)
        p_t = probs * labels + (1 - probs) * (1 - labels)
        modulating_factor = (1.0 - p_t) ** self.gamma
        
        pos_counts = labels.sum(dim=0, keepdim=True)
        neg_counts = (1 - labels).sum(dim=0, keepdim=True)
        total = pos_counts + neg_counts
        
        # Tính class weight động
        alpha_weight = torch.where(
            labels > 0,
            (total / (pos_counts + 1e-5)) * self.alpha,
            (total / (neg_counts + 1e-5)) * (1 - self.alpha)
        )

        focal_loss = alpha_weight * modulating_factor * bce_loss
        masked_loss = (focal_loss * pad_mask).sum() / pad_mask.sum().clamp(min=1e-8)
        return masked_loss

class LossFunctionFactory:
    @staticmethod
    def get_loss(loss_name, **kwargs):
        """
        Factory method để lấy loss function theo tên
        
        Args:
            loss_name: Tên của loss function
            **kwargs: Các tham số bổ sung cho loss function
            
        Returns:
            Một instance của loss function được yêu cầu
        """
        loss_dict = {
            # Single-label losses
            "ce": CrossEntropyLoss,
            "weighted_ce": WeightedCrossEntropyLoss,
            "focal_single": FocalLossForSingleLabel,
            
            # Multi-label losses
            "bce": BinaryCrossEntropyLoss,
            "weighted_bce": WeightedBinaryCrossEntropyLoss,
            "focal_multi": FocalLossForMultiLabel,
            
            # Losses with pad mask
            "bce_with_pad_mask": BCEWithPadMaskLoss,
            "focal_with_pad_mask": FocalLossWithPadMask,
        }
        
        if loss_name not in loss_dict:
            raise ValueError(f"Loss function '{loss_name}' not found. Available losses: {list(loss_dict.keys())}")
        
        return loss_dict[loss_name](**kwargs)