import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        """
        Args:
            gamma (float): focusing parameter.
            alpha (Tensor or list or None): weights for each class.
            reduction (str): 'mean', 'sum', or 'none'.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        if alpha is not None:
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = None
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: logits of shape (N, C)
            targets: ground truth class indices of shape (N)
        """
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        targets = targets.long()

        ce_loss = F.nll_loss(log_probs, targets, reduction='none')
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        if self.alpha is not None:
            alpha_t = self.alpha.to(inputs.device).gather(0, targets)
            focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
