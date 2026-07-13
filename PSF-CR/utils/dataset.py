import os
import torch
from torch.utils.data import Dataset
import numpy as np

def load_image(path):
    """
    Helper function to load satellite image data.
    Supports .npy and .tif files.
    Returns a numpy array of shape (C, H, W).
    """
    if path.endswith('.npy'):
        img = np.load(path)

        if img.ndim == 3 and (img.shape[-1] == 1 or img.shape[-1] == 3 or img.shape[-1] == 4) and img.shape[0] > 4:
            img = np.transpose(img, (2, 0, 1))
        elif img.ndim == 2:
            img = np.expand_dims(img, axis=0)
        return img

    try:
        import rasterio
        with rasterio.open(path) as src:
            img = src.read()
            return img
    except ImportError:
        try:
            import tifffile
            img = tifffile.imread(path)
            if img.ndim == 3 and img.shape[-1] <= 4:
                img = np.transpose(img, (2, 0, 1))
            elif img.ndim == 2:
                img = np.expand_dims(img, axis=0)
            return img
        except ImportError:
            raise ImportError("Please install either 'rasterio' or 'tifffile' to load .tif images.")

class SatDataset(Dataset):
    """
    Dataset loader for 5M sample Kaggle Dataset.
    Loads cloudy, sar, dem, temporal, and clear images.
    """
    def __init__(self, data_dir, transform=None, require_temporal=True):
        """
        data_dir: e.g., 'training_data/' containing subdirectories:
                  clear, cloudy, dem, sar, temporal, mask.
        require_temporal: if False, skips temporal/dem check (used by Approach 1)
        """
        self.data_dir = data_dir
        self.transform = transform
        self.require_temporal = require_temporal

        self.filenames = []

        cloudy_dir = os.path.join(self.data_dir, 'cloudy')
        if not os.path.exists(cloudy_dir):
            print(f"Warning: Directory not found: {cloudy_dir}")
            cloudy_files = []
        else:
            cloudy_files = [f for f in os.listdir(cloudy_dir) if f.endswith(('.tif', '.npy', '.tiff'))]

        for f in sorted(cloudy_files):

            core_ok = (self._get_path('cloudy', f) and
                       self._get_path('clear', f) and
                       self._get_path('sar', f) and
                       self._get_path('mask', f))

            aux_ok = (self._get_path('temporal', f) and
                      self._get_path('dem', f)) if self.require_temporal else True

            if core_ok and aux_ok:
                self.filenames.append(f)

        if not self.filenames:
            print(f"Warning: No valid complete files found in {self.data_dir}. Using dummy data.")
            self.filenames = [f"dummy_{i}.tif" for i in range(10)]

    def _get_path(self, modality, filename):
        p = os.path.join(self.data_dir, modality, filename)
        return p if os.path.exists(p) else None

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]

        p_cloudy = self._get_path('cloudy', filename)
        p_clear = self._get_path('clear', filename)
        p_sar = self._get_path('sar', filename)
        p_dem = self._get_path('dem', filename)
        p_temp = self._get_path('temporal', filename)
        p_mask = self._get_path('mask', filename)

        core_ok = p_cloudy and p_clear and p_sar and p_mask
        aux_ok = (p_temp and p_dem) if self.require_temporal else True

        if core_ok and aux_ok:
            cloudy = load_image(p_cloudy).astype(np.float32)
            clear = load_image(p_clear).astype(np.float32)
            sar = load_image(p_sar).astype(np.float32)
            mask = load_image(p_mask).astype(np.float32)

            if p_temp and self.require_temporal:
                temporal = load_image(p_temp).astype(np.float32)
                temporal = torch.from_numpy(temporal)
                temporal = torch.clamp(temporal, 0.0, 1.0)
            else:
                temporal = torch.zeros((4, cloudy.shape[1], cloudy.shape[2]), dtype=torch.float32)

            if p_dem and self.require_temporal:
                dem = load_image(p_dem).astype(np.float32)
            else:
                dem = np.zeros((1, cloudy.shape[1], cloudy.shape[2]), dtype=np.float32)

            cloudy = torch.from_numpy(cloudy)
            clear = torch.from_numpy(clear)
            sar = torch.from_numpy(sar)
            dem = torch.from_numpy(dem)
            mask = torch.from_numpy(mask)

            opt_scale = 1023.0
            cloudy = torch.clamp(cloudy / opt_scale, 0.0, 1.0)
            clear = torch.clamp(clear / opt_scale, 0.0, 1.0)

            sar_vv = torch.clamp(sar[0:1], -25.0, 0.0) / 25.0 + 1.0
            sar_vh = torch.clamp(sar[1:2], -32.5, 0.0) / 32.5 + 1.0
            sar = torch.cat([sar_vv, sar_vh], dim=0)

            mask = torch.clamp(mask, 0.0, 1.0)

            dem = (dem - dem.mean()) / (dem.std() + 1e-8)
        else:

            s = 256
            cloudy = torch.rand(3, s, s)
            clear = torch.rand(3, s, s)
            sar = torch.rand(2, s, s)
            dem = torch.rand(1, s, s)
            temporal = torch.rand(4, s, s)
            mask = torch.rand(1, s, s)

        sample = {
            'cloudy': cloudy,
            'clear': clear,
            'sar': sar,
            'dem': dem,
            'temporal': temporal,
            'mask': mask
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
