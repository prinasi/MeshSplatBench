"""Loss functions for TriBench training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class L1Loss(nn.Module):
    """L1 (Mean Absolute Error) loss for image reconstruction."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute L1 loss.
        
        Args:
            pred: Predicted image, shape [B, H, W, 3] or [H, W, 3].
            target: Target image, same shape as pred.
            
        Returns:
            Scalar L1 loss.
        """
        return F.l1_loss(pred, target)


class SSIMLoss(nn.Module):
    """Structural Similarity loss using pytorch-msssim."""

    def __init__(self, data_range: float = 1.0):
        super().__init__()
        self.data_range = data_range

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute 1 - SSIM loss.
        
        Args:
            pred: Predicted image, shape [B, H, W, 3] or [B, C, H, W].
            target: Target image, same shape as pred.
            
        Returns:
            Scalar loss (1 - SSIM).
        """
        from pytorch_msssim import ssim

        if pred.dim() == 3:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)

        if pred.shape[-1] in (1, 3, 4):
            pred = pred.permute(0, 3, 1, 2).contiguous()
            target = target.permute(0, 3, 1, 2).contiguous()

        return 1.0 - ssim(pred, target, data_range=self.data_range, size_average=True)


class LPIPSLoss(nn.Module):
    """Learned Perceptual Image Patch Similarity loss."""

    def __init__(self, net: str = "alex"):
        super().__init__()
        import lpips
        self.loss_fn = lpips.LPIPS(net=net, verbose=False)
        self.loss_fn.eval()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute LPIPS loss.
        
        Args:
            pred: Predicted image, shape [B, H, W, 3], values in [0, 1].
            target: Target image, same shape as pred.
            
        Returns:
            Scalar LPIPS loss.
        """
        if pred.dim() == 3:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)

        if pred.shape[-1] in (1, 3, 4):
            pred = pred.permute(0, 3, 1, 2).contiguous()
            target = target.permute(0, 3, 1, 2).contiguous()

        # LPIPS expects [-1, 1] range
        return self.loss_fn(pred * 2 - 1, target * 2 - 1).mean()


class CombinedLoss(nn.Module):
    """Combined loss with configurable weights.
    
    Combines L1 and SSIM losses (and optionally LPIPS) with weights:
        loss = l1_weight * L1 + ssim_weight * (1 - SSIM) [+ lpips_weight * LPIPS]
    """

    def __init__(
        self,
        l1_weight: float = 0.8,
        ssim_weight: float = 0.2,
        lpips_weight: float = 0.0,
    ):
        """Initialize combined loss.
        
        Args:
            l1_weight: Weight for L1 loss.
            ssim_weight: Weight for SSIM loss.
            lpips_weight: Weight for LPIPS loss (0 to disable).
        """
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.lpips_weight = lpips_weight
        self.l1_loss = L1Loss()
        self.ssim_loss = SSIMLoss()
        self.lpips_loss = LPIPSLoss() if lpips_weight > 0 else None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute combined loss.
        
        Returns:
            Dictionary with 'l1', 'ssim', 'total' (and 'lpips' if enabled).
        """
        l1 = self.l1_loss(pred, target)
        ssim = self.ssim_loss(pred, target)
        total = self.l1_weight * l1 + self.ssim_weight * ssim

        result = {"l1": l1, "ssim": ssim, "total": total}

        if self.lpips_loss is not None:
            lpips = self.lpips_loss(pred, target)
            total = total + self.lpips_weight * lpips
            result["lpips"] = lpips
            result["total"] = total

        return result
