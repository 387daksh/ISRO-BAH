import torch
import torch.nn as nn
import torch.nn.functional as F

class Cloudsen12Detector(nn.Module):
    """
    Cloud detector using the CloudSen12 dtacs4bands model, completely ported to PyTorch
    to keep operations fully on the GPU and differentiable/fast for training loops.
    """
    def __init__(self, device='cuda'):
        super().__init__()
        try:
            from cloudsen12_models import cloudsen12
        except ImportError:
            raise ImportError("Please install 'cloudsen12_models' to use the new cloud mask generator.")

        self.model = cloudsen12.load_model_by_name("dtacs4bands").to(device)
        self.model.eval()
        self.device = device

        for param in self.model.parameters():
            param.requires_grad = False

    def _dilate(self, x):
        return F.max_pool2d(x, kernel_size=3, stride=1, padding=1)

    def _erode(self, x):
        return -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)

    def predict_binary(self, x):
        """
        Expects input x to be shape (B, 3, H, W) [Green, Red, NIR].
        Values assumed to be in [0, 1] range.
        Returns a binary mask (0=clear, 1=cloud) of shape (B, 1, H, W).
        """
        B, C, H, W = x.shape

        G = x[:, 0:1, :, :]
        R = x[:, 1:2, :, :]
        NIR = x[:, 2:3, :, :]

        B_band = G

        x_4 = torch.cat([B_band, G, R, NIR], dim=1)

        device_type = (self.device.type if hasattr(self.device, 'type') else str(self.device)).split(':')[0]
        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=torch.float16):
                prediction = self.model(x_4)

        if prediction.ndim == 3:
            prediction = prediction.unsqueeze(1)

        cloud_mask = (prediction > 0.08).float()

        epsilon = 1e-8
        ndvi = (NIR - R) / (NIR + R + epsilon)

        veg_mask = ndvi > 0.50
        soil_mask = R > (G * 1.4)

        dark_mask = (NIR < 0.0586) | (G < 0.0586)

        cloud_mask = cloud_mask.masked_fill(veg_mask, 0.0)
        cloud_mask = cloud_mask.masked_fill(soil_mask, 0.0)
        cloud_mask = cloud_mask.masked_fill(dark_mask, 0.0)

        bright_haze_mask = (G > 0.342) & (R > 0.342) & (NIR > 0.342)
        cloud_mask = torch.logical_or(cloud_mask > 0.5, bright_haze_mask).float()

        m = self._dilate(cloud_mask)
        m = self._dilate(m)
        m = self._erode(m)
        m = self._erode(m)

        for _ in range(5):
            m = self._dilate(m)

        return m
