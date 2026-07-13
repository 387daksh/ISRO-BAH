import torch
import math
from utils.losses import ssim as ssim_fn

def calculate_psnr(pred, target, data_range=1.0):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR).
    Assumes inputs are in range [0, 1]. If not, change data_range.
    """
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return torch.tensor(float('inf'), device=pred.device)
    return 20 * math.log10(data_range) - 10 * torch.log10(mse)

def calculate_sam(pred, target):
    """
    Calculate Spectral Angle Mapper (SAM) across channels.
    Inputs shape: (B, C, H, W)
    Returns scalar: Mean SAM in radians.
    """

    dot_product = torch.sum(pred * target, dim=1)

    norm_pred = torch.norm(pred, dim=1)
    norm_target = torch.norm(target, dim=1)

    cos_theta = dot_product / (norm_pred * norm_target + 1e-8)
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

    sam_map_rad = torch.acos(cos_theta)

    sam_map_deg = sam_map_rad * (180.0 / math.pi)
    return torch.mean(sam_map_deg)

def calculate_ssim(pred, target):
    """
    Wrapper for existing SSIM function.
    """

    return ssim_fn(pred, target)

def compute_all_metrics(pred, target):
    """
    Compute PSNR, SAM, SSIM at once.
    """
    with torch.no_grad():
        psnr = calculate_psnr(pred, target)
        sam = calculate_sam(pred, target)
        ssim = calculate_ssim(pred, target)
    return {'psnr': psnr.item(), 'sam': sam.item(), 'ssim': ssim.item()}
