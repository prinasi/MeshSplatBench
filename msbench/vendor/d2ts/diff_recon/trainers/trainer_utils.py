import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
import cv2
import numpy as np
from pathlib import Path

from msbench.vendor._cmod.simple_knn import nearestNeighbor

from ..utils.camera import Camera
from ..utils.vis_utils import save_image_tensor


class VideoLogger:
    def __init__(self, cameras: list[Camera], save_dir: str):
        self.cameras = cameras
        self.save_dir = save_dir
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        self.video_writers = []
        for i, camera in enumerate(cameras):
            name = camera.image_name if camera.image_name is not None else f"{i:05d}"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(f"{save_dir}/{name}.mp4", fourcc, 30.0, (camera.image_width, camera.image_height))
            self.video_writers.append(video_writer)

    def log(self, model):
        for i, camera in enumerate(self.cameras):
            img = model.forward(camera.to(model.device), "black", False, False)["render"]
            img = (img.detach().cpu().numpy().transpose(1, 2, 0).clip(0, 1) * 255 + 0.5).astype(np.uint8)[:, :, ::-1]
            self.video_writers[i].write(img)

    def close(self):
        for video_writer in self.video_writers:
            video_writer.release()


class GaussianSmoothing2D(nn.Module):
    def __init__(self, kernel_size: int, sigma: float, normalize: bool = True):
        super().__init__()
        self.kernel_size = kernel_size
        self.kernel = self._gaussian_kernel(kernel_size, sigma, normalize)
        self.kernel_buffer = {}

    @staticmethod
    def _gaussian_kernel(kernel_size: int, sigma: float, normalize: bool):
        x_grid = torch.arange(kernel_size).unsqueeze(0).repeat(kernel_size, 1)
        xy_grid = torch.stack([x_grid, x_grid.T], dim=-1).float()

        mean = (kernel_size - 1) / 2.0
        variance = sigma**2.0

        kernel = torch.exp(-(xy_grid - mean).pow(2).sum(dim=-1) / (2 * variance))
        if normalize:
            kernel = kernel / kernel.sum()
        kernel = kernel.unsqueeze(0).unsqueeze(0).float()

        return kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Tensor of shape (B, C, H, W)
        """
        channels = x.shape[1]
        if channels not in self.kernel_buffer:
            self.kernel_buffer[channels] = self.kernel.repeat(channels, 1, 1, 1).to(x.device)

        kernel = self.kernel_buffer[channels].to(x.device)
        padding = (self.kernel_size - 1) // 2
        return F.conv2d(x, kernel, padding=padding, groups=channels)


class SSIM(nn.Module):
    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.window = GaussianSmoothing2D(window_size, sigma)
        self.C1 = 0.01**2
        self.C2 = 0.03**2

    def forward(self, img1: torch.Tensor, img2: torch.Tensor, keep_batch_dim: bool = False) -> torch.Tensor:
        """
        Args:
            img1 (torch.Tensor): Tensor of shape (B, C, H, W)
            img2 (torch.Tensor): Tensor of shape (B, C, H, W)
            keep_batch_dim (bool): Whether to keep the batch dimension or not.
        """
        mu1 = self.window(img1)
        mu2 = self.window(img2)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = self.window(img1 * img1) - mu1_sq
        sigma2_sq = self.window(img2 * img2) - mu2_sq
        sigma12 = self.window(img1 * img2) - mu1_mu2

        C1 = self.C1
        C2 = self.C2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        ssim_val = ssim_map.mean(dim=(1, 2, 3)) if keep_batch_dim else ssim_map.mean()
        return ssim_val


def normalize_shape(*imgs) -> tuple[torch.Tensor, ...]:
    # Normalize input image shapes to (B, C, H, W)
    normalized_imgs = []
    for img in imgs:
        if img is None:
            pass
        elif not torch.is_tensor(img):
            raise ValueError("Input images must be torch tensors.")
        elif len(img.size()) not in [2, 3, 4]:
            raise ValueError("Input images must have 2, 3, or 4 dimensions.")
        elif len(img.size()) == 2:
            img = img.unsqueeze(0).unsqueeze(0)
        elif len(img.size()) == 3:
            img = img.unsqueeze(0)
        elif len(img.size()) == 4:
            pass
        normalized_imgs.append(img)
    return tuple(normalized_imgs) if len(normalized_imgs) > 1 else normalized_imgs[0]


class SSIMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ssim = SSIM()

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        img1, img2 = normalize_shape(img1, img2)
        return 1 - self.ssim(img1, img2)


class DoGFilter(nn.Module):
    def __init__(self, sigma: float):
        super().__init__()
        self.sigma1 = sigma
        self.sigma2 = 2 * sigma  # Ensure the 1:2 ratio
        self.kernel_size1 = int(2 * round(3 * self.sigma1) + 1)
        self.kernel_size2 = int(2 * round(3 * self.sigma2) + 1)
        self.filter1 = GaussianSmoothing2D(self.kernel_size1, self.sigma1)
        self.filter2 = GaussianSmoothing2D(self.kernel_size2, self.sigma2)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gaussian1 = self.filter1(x)
        gaussian2 = self.filter2(x)
        dog = gaussian1 - gaussian2
        return dog


class DoGLoss(nn.Module):
    def __init__(self, freq: int = 90, scale_factor: float = 0.5):
        super().__init__()
        self.freq = freq
        self.scale_factor = scale_factor
        self.dog_filter = DoGFilter(0.1 + (100 - freq) * 0.1 if freq >= 50 else 0.1 + freq * 0.1)

    @torch.no_grad()
    def _dog_mask(self, img: torch.Tensor) -> torch.Tensor:
        grayscale = img.mean(dim=1, keepdim=True)
        downsampled = F.interpolate(grayscale, scale_factor=self.scale_factor, mode="bilinear", align_corners=False)
        dog_img = self.dog_filter(downsampled)
        upsampled = F.interpolate(dog_img, size=img.shape[-2:], mode="bilinear", align_corners=False)

        normalized = (upsampled - upsampled.min()) / (upsampled.max() - upsampled.min())
        if self.freq >= 50:
            normalized = 1.0 - normalized

        mask = (normalized >= 0.5).float()
        return mask

    def forward(self, img: torch.Tensor, img_gt: torch.Tensor) -> torch.Tensor:
        img, img_gt = normalize_shape(img, img_gt)
        mask = self._dog_mask(img_gt)
        return L1(img * mask, img_gt * mask)


class ScharrFilter(nn.Module):
    def __init__(self):
        super().__init__()
        self.kernel_x = torch.tensor([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=torch.float32).view(1, 1, 3, 3) / 32
        self.kernel_y = torch.tensor([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=torch.float32).view(1, 1, 3, 3) / 32
        self.kernel_buffer = {}

    def forward(self, x: torch.Tensor, ret_norm=False) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Tensor of shape (B, C, H, W)
        """
        channels = x.shape[1]
        if channels not in self.kernel_buffer:
            self.kernel_buffer[channels] = (
                self.kernel_x.repeat(channels, 1, 1, 1).to(x.device),
                self.kernel_y.repeat(channels, 1, 1, 1).to(x.device),
            )

        kernel_x, kernel_y = self.kernel_buffer[channels]
        x_padded = F.pad(x, (1, 1, 1, 1), mode="replicate")
        grad_x = F.conv2d(x_padded, kernel_x, groups=channels)
        grad_y = F.conv2d(x_padded, kernel_y, groups=channels)
        grad = torch.cat((grad_x, grad_y), dim=1)

        if ret_norm:
            grad = grad.norm(dim=1, keepdim=True)
        return grad


class SmoothnessLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.scharr = ScharrFilter()

    def forward(
        self, img: torch.Tensor, img_gt: torch.Tensor = None, alpha_mask: torch.Tensor = None, scale_factor: float = 1.0, quantile: float = 1.0
    ) -> torch.Tensor:
        img, img_gt, alpha_mask = normalize_shape(img, img_gt, alpha_mask)

        if scale_factor != 1:
            img = F.interpolate(img, scale_factor=scale_factor, mode="bilinear", align_corners=False)
            if img_gt is not None:
                img_gt = F.interpolate(img_gt, scale_factor=scale_factor, mode="bilinear", align_corners=False)
            if alpha_mask is not None:
                alpha_mask = F.interpolate(alpha_mask, scale_factor=scale_factor, mode="bilinear", align_corners=False)

        grad_img = self.scharr(img, ret_norm=True)
        if quantile is None or quantile >= 1.0:
            grad_mask = 1.0
        else:
            grad_threshold = torch.quantile(grad_img, quantile)
            grad_mask = (grad_img < grad_threshold).float().detach()

        if img_gt is None:
            img_grad_weight = 1.0
        else:
            img_grad = self.scharr(img_gt, ret_norm=True)
            # img_grad_weight = torch.exp(-img_grad * 10).detach()
            img_grad_weight = ((img_grad.max() - img_grad) / (img_grad.max() - img_grad.min() + 1e-10)).detach() ** 2

        if alpha_mask is None:
            alpha_mask = 1.0
        else:
            alpha_mask = -F.max_pool2d(-alpha_mask, kernel_size=5, stride=1, padding=2).detach()  # erode mask to avoid boundary artifacts

        loss = (grad_img * grad_mask * img_grad_weight * alpha_mask).mean()
        return loss


class DepthNormalLoss(nn.Module):
    def __init__(self, depth_grad: bool = True, normal_grad: bool = True, scale_factor: float = None, depth_grad_filter_quantile: float = 1.0):
        super().__init__()
        self.depth_grad = depth_grad
        self.normal_grad = normal_grad
        self.scale_factor = scale_factor
        self.depth_grad_filter_quantile = depth_grad_filter_quantile
        self.scharr = ScharrFilter()

    def depth_to_normal(self, depth: torch.Tensor, tan_fovx: float, tan_fovy: float, ret_grad_mask: bool = True) -> torch.Tensor:
        """
        Args:
            depth (torch.Tensor): Tensor of shape (H, W)
        """
        scale_factor = self.scale_factor
        W0, H0 = depth.shape[-1], depth.shape[-2]
        depth = depth.unsqueeze(0).unsqueeze(0)
        if scale_factor is not None and scale_factor != 1:
            depth = F.interpolate(depth, scale_factor=scale_factor, mode="bilinear", align_corners=False)

        depth_grad = self.scharr(depth).squeeze(0)
        Dx, Dy = torch.unbind(depth_grad, 0)
        W, H = depth.shape[-1], depth.shape[-2]
        x = torch.arange(W, dtype=torch.float32).to(depth.device)
        y = torch.arange(H, dtype=torch.float32).to(depth.device)
        x, y = torch.meshgrid(x, y, indexing="xy")

        nx = W * Dx / (2 * tan_fovx)
        ny = H * Dy / (2 * tan_fovy)
        nz = -(depth.squeeze(0).squeeze(0) + (x - W / 2 + 0.5) * Dx + (y - H / 2 + 0.5) * Dy)
        normal = torch.stack([nx, ny, nz], dim=0)

        if W0 != W or H0 != H:
            normal = F.interpolate(normal.unsqueeze(0), size=(H0, W0), mode="bilinear", align_corners=False).squeeze(0)
        normal = F.normalize(normal, p=2, dim=0, eps=1e-8)
        if not ret_grad_mask:
            return normal

        if self.depth_grad_filter_quantile is None or self.depth_grad_filter_quantile >= 1.0:
            grad_mask = torch.ones((H0, W0), device=depth.device)
        else:
            grad_norm = depth_grad.norm(dim=0, keepdim=True)
            if W0 != W or H0 != H:
                grad_norm = F.interpolate(grad_norm.unsqueeze(0), size=(H0, W0), mode="bilinear", align_corners=False).squeeze(0)
            grad_threshold = torch.quantile(grad_norm, self.depth_grad_filter_quantile)
            grad_mask = (grad_norm < grad_threshold).float().squeeze(0).detach()
        return normal, grad_mask

    def forward(
        self,
        depth: torch.Tensor,
        normal: torch.Tensor,
        tan_fovx: float,
        tan_fovy: float,
        alpha_mask: torch.Tensor = None,
        img_gt: torch.Tensor = None,
    ) -> torch.Tensor:
        if not self.depth_grad:
            depth = depth.detach()
        if not self.normal_grad:
            normal = normal.detach()

        depth_normal, grad_mask = self.depth_to_normal(depth, tan_fovx, tan_fovy)
        # normal = torch.nn.functional.normalize(normal, p=2, dim=0, eps=1e-8)
        img_gt, alpha_mask = normalize_shape(img_gt, alpha_mask)

        if img_gt is None:
            img_grad_weight = 1.0
        else:
            img_grad = self.scharr(img_gt, ret_norm=True)
            # img_grad_weight = torch.exp(-img_grad * 10).detach()
            img_grad_weight = ((img_grad.max() - img_grad) / (img_grad.max() - img_grad.min() + 1e-10)).detach() ** 2

        if alpha_mask is None:
            alpha_mask = 1.0
        else:
            alpha_mask = -F.max_pool2d(-alpha_mask, kernel_size=5, stride=1, padding=2).detach()  # erode mask to avoid boundary artifacts

        loss_img = 1 - (normal * depth_normal).sum(dim=0)
        loss = (loss_img * grad_mask * img_grad_weight * alpha_mask).mean()
        return loss


def lncc(patch1: torch.Tensor, patch2: torch.Tensor) -> torch.Tensor:
    """
    Local normalized cross-correlation loss
    Args:
        patch1 (torch.Tensor): Reference tensor of shape (N, K)
        patch2 (torch.Tensor): Neighbor tensor of shape (N, K)
    Returns:
        ncc (torch.Tensor): NCC loss of shape (N,)
    """
    if not patch1.shape == patch2.shape:
        raise ValueError("Input patches must have the same shape.")

    tps = patch1.shape[1]
    patch1_sum = patch1.sum(dim=1)
    patch2_sum = patch2.sum(dim=1)
    patch1_sq_sum = (patch1 * patch1).sum(dim=1)
    patch2_sq_sum = (patch2 * patch2).sum(dim=1)
    patch1_patch2_sum = (patch1 * patch2).sum(dim=1)

    patch1_var = patch1_sq_sum - patch1_sum * patch1_sum / tps
    patch2_var = patch2_sq_sum - patch2_sum * patch2_sum / tps
    cross = patch1_patch2_sum - patch1_sum * patch2_sum / tps
    cc = cross * cross / (patch1_var * patch2_var + 1e-8)

    ncc = (1 - cc).clamp(0.0, 2.0)
    return ncc


class ConsistencyLoss(nn.Module):
    def __init__(self, error_thres: float = 1.0, n_sample: int = None, patch_size: int = 3, dilation: int = 1):
        super().__init__()
        self.error_thres = error_thres
        self.n_sample = n_sample
        self.patch_size = patch_size
        self.dilation = dilation

        patch_offset = torch.arange(-patch_size, patch_size + 1) * dilation
        patch_offset = torch.stack(torch.meshgrid(patch_offset, patch_offset, indexing="xy"), dim=-1).view(-1, 2)  # (K, 2)
        self.patch_offset = patch_offset

    def forward(self, model, render_pkg: dict, render_pkg_ref: dict) -> tuple[torch.Tensor, torch.Tensor]:
        camera: Camera = render_pkg["camera"]
        camera_ref: Camera = render_pkg_ref["camera"]
        depth: torch.Tensor = render_pkg["depth"]
        depth_ref: torch.Tensor = render_pkg_ref["depth"]
        gt_img: torch.Tensor = camera.gt_image
        gt_img_ref: torch.Tensor = camera_ref.gt_image
        device = camera.device
        wh = torch.tensor([camera.image_width, camera.image_height], device=device)

        # project points from current view to reference view
        xyz, color, normal, pixels = model.render_points(camera, render_pkg, ret_pointcloud=False)
        if xyz.shape[0] == 0:
            return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
        xyz_proj = camera_ref.project_points(xyz)

        # filter out points that are outside the view frustum
        frustum_mask = (xyz_proj[:, 0] > -1) & (xyz_proj[:, 0] < 1) & (xyz_proj[:, 1] > -1) & (xyz_proj[:, 1] < 1) & (xyz_proj[:, 2] > 0)
        # print("In frustum:", frustum_mask.float().mean().item() * 100.0, "%")
        if not frustum_mask.any():
            return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
        xyz_proj = xyz_proj[frustum_mask]
        pixels = pixels[frustum_mask]

        # sample points from reference view and project back to current view
        xyz_ref = camera_ref.get_xyz_from_depth(depth_ref).unsqueeze(0).permute(0, 3, 1, 2)  # (1, 3, H, W)
        xyz_ref_sampled = F.grid_sample(xyz_ref, xyz_proj[:, :2].view(1, -1, 1, 2), align_corners=False).view(-1, 3)  # (N, 3)
        xy_proj_back = camera.project_points(xyz_ref_sampled)[:, :2]  # (N, 2)
        pixels_back = (xy_proj_back + 1) * 0.5 * wh

        # compute geometry loss
        error = (pixels - pixels_back).norm(dim=1)  # in pixel space
        error_mask = (error < self.error_thres).detach()
        # print("Consistent pixels:", error_mask.float().mean().item() * 100.0, "%")
        if not error_mask.any():
            return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
        error_weight = torch.exp(-error).detach() * error_mask.float()
        geo_loss = (error_weight * error)[error_mask].mean()

        if gt_img is None or gt_img_ref is None:  # cannot compute color loss without ground truth images
            return geo_loss, torch.tensor(0.0, device=device)

        # generate pixel patches on current view for color loss computation
        sampled_idx = torch.randperm(error_mask.sum())[: self.n_sample] if self.n_sample is not None else torch.arange(error_mask.sum())
        sampled_pixels = pixels[error_mask][sampled_idx]
        sampled_error_weight = error_weight[error_mask][sampled_idx]
        pixel_patch = sampled_pixels.unsqueeze(1) + self.patch_offset.unsqueeze(0).to(device)  # (N, K, 2)
        pixel_patch = (pixel_patch / wh * 2) - 1  # normalize to [-1, 1]

        # sample colors from current view
        gt_img_gray = (0.299 * gt_img[0] + 0.587 * gt_img[1] + 0.114 * gt_img[2]).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        patch_color_gt = F.grid_sample(gt_img_gray, pixel_patch.view(1, -1, 1, 2), align_corners=False).view(pixel_patch.shape[0], -1)  # (N, K)

        # project patch points to reference view
        xyz_img = camera.get_xyz_from_depth(depth).unsqueeze(0).permute(0, 3, 1, 2)  # (1, 3, H, W)
        patch_xyz = F.grid_sample(xyz_img, pixel_patch.view(1, -1, 1, 2), align_corners=False).view(-1, 3)  # (N*K, 3)
        patch_proj = camera_ref.project_points(patch_xyz)[:, :2]  # (N*K, 2)

        # sample colors from reference view
        gt_img_ref_gray = (0.299 * gt_img_ref[0] + 0.587 * gt_img_ref[1] + 0.114 * gt_img_ref[2]).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        patch_color_sampled = F.grid_sample(gt_img_ref_gray, patch_proj.view(1, -1, 1, 2), align_corners=False).view(pixel_patch.shape[0], -1)  # (N, K)

        # compute color loss
        ncc = lncc(patch_color_gt, patch_color_sampled)
        ncc_mask = (ncc < 0.9).detach()
        # print("Consistent colors:", ncc_mask.float().mean().item() * 100.0, "%")
        if not ncc_mask.any():
            return geo_loss, torch.tensor(0.0, device=device)
        color_loss = (sampled_error_weight * ncc)[ncc_mask].mean()

        return geo_loss, color_loss


def L1(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    return torch.abs((t1 - t2)).mean()


def L2(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    return ((t1 - t2) ** 2).mean()


def psnr(img1: torch.Tensor, img2: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    if mask is None:
        mse = ((img1 - img2) ** 2).mean() + 1e-10
    else:
        mse = (((img1 - img2) ** 2) * mask).sum() / (mask.sum() + 1e-10) + 1e-10
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def nearest_neighbor(pc: torch.Tensor, bs: int = 1) -> torch.Tensor:
    return nearestNeighbor(pc, bs).long()


def nearest_dist2(pc: torch.Tensor, nearest_indices: torch.Tensor) -> torch.Tensor:
    assert pc.dim() == 2 and pc.size(1) == 3 and nearest_indices.dim() == 1 and nearest_indices.size(0) == pc.size(0)
    nearest_point = pc[nearest_indices]
    return ((pc - nearest_point) ** 2).sum(dim=1)


ssimLoss = SSIMLoss()
dogLoss = DoGLoss()
smoothnessLoss = SmoothnessLoss()
