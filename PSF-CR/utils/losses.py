import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def gaussian_window(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
                          for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D = gaussian_window(window_size, 1.5).unsqueeze(1)
    _2D = _1D.mm(_1D.t()).float().unsqueeze(0).unsqueeze(0)
    return _2D.expand(channel, 1, window_size, window_size).contiguous()

def ssim(img1, img2, window_size=11):
    channel = img1.size(1)
    window = create_window(window_size, channel).to(img1.device)
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    s1 = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    s2 = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    s12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    return ((2 * mu1_mu2 + C1) * (2 * s12 + C2) /
            ((mu1_sq + mu2_sq + C1) * (s1 + s2 + C2))).mean()

class SpatialLoss(nn.Module):
    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred, target, cloud_mask):

        k = 2.0
        weight = 1.0 + (k - 1.0) * cloud_mask

        l1_loss = torch.mean(weight * torch.abs(pred - target))

        ssim_loss = 1.0 - ssim(pred, target)

        return l1_loss + self.alpha * ssim_loss

class FrequencyLoss(nn.Module):
    def forward(self, pred, target):
        pred_f = torch.fft.fft2(pred.float(), norm='ortho')
        target_f = torch.fft.fft2(target.float(), norm='ortho')
        real_loss = torch.mean(torch.abs(pred_f.real - target_f.real))
        imag_loss = torch.mean(torch.abs(pred_f.imag - target_f.imag))
        diff_norm = torch.linalg.matrix_norm(pred_f - target_f, ord='fro', dim=(-2, -1))
        target_norm = torch.linalg.matrix_norm(target_f, ord='fro', dim=(-2, -1))
        sc_loss = torch.mean(diff_norm / (target_norm + 1e-4))
        return real_loss + imag_loss + sc_loss

class PSFCRLoss(nn.Module):
    """Original PSF-CR paper loss. Used for Phase 1 pre-training."""
    def __init__(self, lambda1=1.0, lambda2=0.1):
        super().__init__()
        self.spatial_loss = SpatialLoss()
        self.freq_loss = FrequencyLoss()
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def forward(self, pred, target, w_matrix):
        return self.lambda1 * self.spatial_loss(pred, target, w_matrix) + \
               self.lambda2 * self.freq_loss(pred, target)

class NDVILoss(nn.Module):
    """
    L1 loss on NDVI maps.

    WHY THIS IS NEEDED FOR LISS-IV:
    LISS-IV captures R, G, NIR — the three bands that define NDVI
    (Normalized Difference Vegetation Index = (NIR - R) / (NIR + R)).
    NDVI is the single most critical spectral index for Northeast India's
    applications (crop health, flood extent mapping, forest cover for ISRO).

    The standard L1 loss treats all pixel differences equally. A pixel with
    a completely wrong NDVI value in a dense paddy field gets the same
    penalty as a wrong pixel on a concrete road. This is incorrect — in
    Northeast India's landscape, the vegetation response is the primary
    information content.

    The NDVILoss directly penalizes errors in vegetation index space,
    forcing the model to recover ecologically and spectrally correct
    vegetation signatures even inside the cloud-covered areas.
    It is weighted by the cloud mask so it only penalizes the reconstructed
    region (inside the cloud) where temporal change could corrupt the NDVI.

    Input tensors must be [Green, Red, NIR] at channel indices [0, 1, 2].
    """

    def forward(self, pred, target, w_matrix):

        p = torch.clamp(pred, 0.0, 1.0)
        t = torch.clamp(target, 0.0, 1.0)

        pred_ndvi   = (p[:, 2:3] - p[:, 1:2]) / (p[:, 2:3] + p[:, 1:2] + 1e-3)
        target_ndvi = (t[:, 2:3] - t[:, 1:2]) / (t[:, 2:3] + t[:, 1:2] + 1e-3)

        masked_loss = torch.mean(w_matrix * torch.abs(pred_ndvi - target_ndvi))
        return masked_loss

class Phase2Loss(nn.Module):
    """
    Combined loss for Phase 2 LISS-IV fine-tuning.
    Extends PSFCRLoss with NDVILoss.

    lambda1 : weight for spatial loss (L1 + SSIM)
    lambda2 : weight for frequency loss (Spectral Convergence, cloud-masked)
    lambda3 : weight for NDVI loss — recommended 0.5 to 1.0
               (lower than spatial loss since NDVI is derived from pred channels
                which are already penalized by L1; it acts as a specialist term)
    """
    def __init__(self, lambda1=1.0, lambda2=0.5, lambda3=0.0):
        super().__init__()

        self.spatial_loss = SpatialLoss(alpha=1.5)
        self.freq_loss    = FrequencyLoss()
        self.ndvi_loss    = NDVILoss()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3

    def forward(self, pred, target, w_matrix):
        if w_matrix.size(1) > 1:
            w_matrix = w_matrix.mean(dim=1, keepdim=True)

        L_spa  = self.spatial_loss(pred, target, w_matrix)

        L_freq = self.freq_loss(pred, target)

        L_ndvi = self.ndvi_loss(pred, target, w_matrix)

        return self.lambda1 * L_spa + self.lambda2 * L_freq + self.lambda3 * L_ndvi
