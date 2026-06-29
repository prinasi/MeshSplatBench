"""Evaluation metrics for rendered image quality."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute Peak Signal-to-Noise Ratio.
    
    Args:
        pred: Predicted image, shape [H, W, 3] or [B, H, W, 3], values in [0, 1].
        target: Target image, same shape as pred, values in [0, 1].
        
    Returns:
        PSNR value in dB. Returns float('inf') if images are identical.
    """
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    mse = F.mse_loss(pred, target, reduction="mean")
    if mse < 1e-10:
        return float("inf")
    return 10.0 * torch.log10(1.0 / mse).item()


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
) -> float:
    """Compute Structural Similarity Index.
    
    Uses pytorch-msssim for efficient GPU computation.
    
    Args:
        pred: Predicted image, shape [H, W, 3] or [B, H, W, 3], values in [0, 1].
        target: Target image, same shape as pred, values in [0, 1].
        data_range: Value range of the input images.
        
    Returns:
        SSIM value in [0, 1].
    """
    from pytorch_msssim import ssim

    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    # pytorch_msssim expects [B, C, H, W]
    pred_chw = pred.permute(0, 3, 1, 2).contiguous()
    target_chw = target.permute(0, 3, 1, 2).contiguous()

    return ssim(pred_chw, target_chw, data_range=data_range, size_average=True).item()


def compute_lpips(
    pred: torch.Tensor,
    target: torch.Tensor,
    net: str = "vgg",
) -> float:
    """Compute Learned Perceptual Image Patch Similarity.
    
    Args:
        pred: Predicted image, shape [H, W, 3] or [B, H, W, 3], values in [0, 1].
        target: Target image, same shape as pred, values in [0, 1].
        net: Backbone network ('alex', 'vgg', 'squeeze').
        
    Returns:
        LPIPS value (lower is better).
    """
    import lpips as lpips_lib

    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    # lpips expects [B, C, H, W] in range [-1, 1]
    pred_chw = pred.permute(0, 3, 1, 2).contiguous() * 2.0 - 1.0
    target_chw = target.permute(0, 3, 1, 2).contiguous() * 2.0 - 1.0

    loss_fn = lpips_lib.LPIPS(net=net, verbose=False).to(pred.device)
    with torch.no_grad():
        result = loss_fn(pred_chw, target_chw).mean().item()
    return result


def compute_all_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    """Compute PSNR, SSIM, and LPIPS at once.

    Args:
        pred: Predicted image, shape [H, W, 3] or [B, H, W, 3], values in [0, 1].
        target: Target image, same shape as pred, values in [0, 1].

    Returns:
        Dictionary with keys 'psnr', 'ssim', 'lpips'.
    """
    return {
        "psnr": compute_psnr(pred, target),
        "ssim": compute_ssim(pred, target),
        "lpips": compute_lpips(pred, target),
    }


def compute_masked_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """Compute PSNR only over foreground pixels (where mask > 0.5).

    Args:
        pred: Predicted image, shape [H, W, 3], values in [0, 1].
        target: Target image, same shape as pred, values in [0, 1].
        mask: Binary mask, shape [H, W, 1] or [H, W], values in [0, 1].

    Returns:
        PSNR value in dB computed over foreground pixels only.
    """
    if mask.dim() == 3 and mask.shape[-1] == 1:
        mask = mask.squeeze(-1)
    fg = mask > 0.5
    pred_fg = pred[fg]
    target_fg = target[fg]
    if pred_fg.numel() == 0:
        return float("inf")
    mse = F.mse_loss(pred_fg, target_fg, reduction="mean")
    if mse < 1e-10:
        return float("inf")
    return 10.0 * torch.log10(1.0 / mse).item()


def compute_masked_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    data_range: float = 1.0,
) -> float:
    """Compute SSIM on mask-multiplied images.

    Both pred and target are multiplied by the mask (zeroing background)
    before SSIM computation.

    Args:
        pred: Predicted image, shape [H, W, 3], values in [0, 1].
        target: Target image, same shape as pred, values in [0, 1].
        mask: Mask, shape [H, W, 1] or [H, W], values in [0, 1].
        data_range: Value range of the input images.

    Returns:
        SSIM value.
    """
    from pytorch_msssim import ssim

    if mask.dim() == 2:
        mask = mask.unsqueeze(-1)
    pred_masked = (pred * mask).unsqueeze(0)
    target_masked = (target * mask).unsqueeze(0)
    pred_chw = pred_masked.permute(0, 3, 1, 2).contiguous()
    target_chw = target_masked.permute(0, 3, 1, 2).contiguous()
    return ssim(pred_chw, target_chw, data_range=data_range, size_average=True).item()


def compute_masked_lpips(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    net: str = "vgg",
) -> float:
    """Compute LPIPS on mask-multiplied images.

    Both pred and target are multiplied by the mask before LPIPS computation.

    Args:
        pred: Predicted image, shape [H, W, 3], values in [0, 1].
        target: Target image, same shape as pred, values in [0, 1].
        mask: Mask, shape [H, W, 1] or [H, W], values in [0, 1].
        net: Backbone network.

    Returns:
        LPIPS value (lower is better).
    """
    import lpips as lpips_lib

    if mask.dim() == 2:
        mask = mask.unsqueeze(-1)
    pred_masked = (pred * mask).unsqueeze(0)
    target_masked = (target * mask).unsqueeze(0)
    pred_chw = pred_masked.permute(0, 3, 1, 2).contiguous() * 2.0 - 1.0
    target_chw = target_masked.permute(0, 3, 1, 2).contiguous() * 2.0 - 1.0
    loss_fn = lpips_lib.LPIPS(net=net, verbose=False).to(pred.device)
    with torch.no_grad():
        result = loss_fn(pred_chw, target_chw).mean().item()
    return result


def compute_all_metrics_with_mask(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> dict[str, float]:
    """Compute full-image metrics and optionally foreground-masked metrics.

    Args:
        pred: Predicted image, shape [H, W, 3], values in [0, 1].
        target: Target image, same shape as pred, values in [0, 1].
        mask: Optional foreground mask, shape [H, W, 1] or [H, W].

    Returns:
        Dictionary with keys 'psnr', 'ssim', 'lpips' (full image).
        When mask is provided, also includes 'psnr_fg', 'ssim_fg', 'lpips_fg'.
    """
    result = compute_all_metrics(pred, target)
    if mask is not None:
        result["psnr_fg"] = compute_masked_psnr(pred, target, mask)
        result["ssim_fg"] = compute_masked_ssim(pred, target, mask)
        result["lpips_fg"] = compute_masked_lpips(pred, target, mask)
    return result


def compute_foreground_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, float]:
    """Compute foreground metrics as the primary PSNR/SSIM/LPIPS values."""
    return {
        "psnr": compute_masked_psnr(pred, target, mask),
        "ssim": compute_masked_ssim(pred, target, mask),
        "lpips": compute_masked_lpips(pred, target, mask),
    }


def evaluate_dataset(
    renderer: Any,
    dataset: Any,
    device: torch.device | str = "cuda",
) -> dict[str, Any]:
    """Evaluate a renderer on a full dataset split.
    
    Renders all views and computes aggregated metrics.
    
    Args:
        renderer: A RendererAdapter instance with loaded model.
        dataset: A DatasetBase instance.
        device: Device for rendering.
        
    Returns:
        Dictionary with per-view and aggregated metrics.
    """
    from tribench.core.cameras import CameraBatch

    per_view_metrics = []

    for i in range(len(dataset)):
        cameras, gt_image = dataset[i]
        cameras = cameras.to(device)
        gt_image = gt_image.to(device)

        output = renderer.render(cameras, mode="eval")
        pred_rgb = output.rgb.clamp(0, 1)

        if gt_image.dim() == 3 and gt_image.shape[-1] == 4:
            gt_image = gt_image[..., :3]

        view_metrics = compute_all_metrics(pred_rgb, gt_image)
        per_view_metrics.append(view_metrics)

    # Aggregate
    n = len(per_view_metrics)
    aggregated = {}
    for key in per_view_metrics[0]:
        values = [m[key] for m in per_view_metrics]
        if all(v != float("inf") for v in values):
            aggregated[f"{key}_mean"] = sum(values) / n
            aggregated[f"{key}_std"] = (
                sum((v - aggregated[f"{key}_mean"]) ** 2 for v in values) / n
            ) ** 0.5
        else:
            aggregated[f"{key}_mean"] = float("inf")
            aggregated[f"{key}_std"] = 0.0

    return {
        "per_view": per_view_metrics,
        "aggregated": aggregated,
        "num_views": n,
    }
