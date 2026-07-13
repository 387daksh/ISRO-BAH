import os
import torch
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import numpy as np
import warnings
import tifffile

warnings.filterwarnings("ignore")
from utils.cloud_detection import Cloudsen12Detector

class CloudPrecomputeDataset(Dataset):
    def __init__(self, data_root):
        self.data_root = Path(data_root)
        self.cloudy_files = []

        cloudy_dir = self.data_root / 'cloudy'
        mask_dir = self.data_root / 'mask'

        if not cloudy_dir.exists():
            return

        mask_dir.mkdir(parents=True, exist_ok=True)

        for cloudy_file in cloudy_dir.iterdir():
            if not cloudy_file.name.endswith(('.npy', '.tif', '.tiff')):
                continue

            mask_file = mask_dir / cloudy_file.name
            if not mask_file.exists():
                self.cloudy_files.append((cloudy_file, mask_file))

    def __len__(self):
        return len(self.cloudy_files)

    def __getitem__(self, idx):
        cloudy_path, mask_path = self.cloudy_files[idx]

        path_str = str(cloudy_path)
        if path_str.endswith('.npy'):
            img = np.load(path_str)
            if img.ndim == 3 and img.shape[-1] <= 4:
                img = np.transpose(img, (2, 0, 1))
            elif img.ndim == 2:
                img = np.expand_dims(img, axis=0)
        else:
            img = tifffile.imread(path_str)
            if img.ndim == 3 and img.shape[-1] <= 4:
                img = np.transpose(img, (2, 0, 1))
            elif img.ndim == 2:
                img = np.expand_dims(img, axis=0)

        img = img.astype(np.float32)
        opt_scale = 1023.0
        img = np.clip(img / opt_scale, 0.0, 1.0)

        return torch.from_numpy(img), str(mask_path)

def precompute_masks(data_root, batch_size=128):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Initializing CloudSen12 for precomputation on {device}...")

    detector = Cloudsen12Detector(device=device)
    dataset = CloudPrecomputeDataset(data_root)

    if len(dataset) == 0:
        print("✅ All masks are already precomputed! Skipping.")
        return False

    print(f"Found {len(dataset)} cloudy patches missing masks. Precomputing now...")

    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    with torch.no_grad():
        for imgs, mask_paths in tqdm(loader, desc="Precomputing Masks"):
            imgs = imgs.to(device)

            masks = detector.predict_binary(imgs)
            masks_np = masks.squeeze(1).byte().cpu().numpy()

            for i in range(len(mask_paths)):
                mask_path = mask_paths[i]
                mask_img = masks_np[i]

                if mask_path.endswith('.npy'):
                    np.save(mask_path, mask_img)
                else:
                    tifffile.imwrite(mask_path, mask_img, compression='deflate')

    return True

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True, help="Path to dataset root")
    args = parser.parse_args()

    precompute_masks(args.data_root)
