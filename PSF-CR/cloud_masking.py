import numpy as np
import rasterio
from rasterio.windows import Window
from cloudsen12_models import cloudsen12
import torch
from tqdm import tqdm
import scipy.ndimage as ndimage

red_path = "R2F06JAN2026076370011100055SSANSTUC00GTDB/BAND3.tif"
green_path = "R2F06JAN2026076370011100055SSANSTUC00GTDB/BAND2.tif"
nir_path = "R2F06JAN2026076370011100055SSANSTUC00GTDB/BAND4.tif"
output_path = "cloud_prediction_FULL.tif"

print("Initializing environment...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using compute device: {device}")

if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True

print("Loading dtacs4bands model...")
model = cloudsen12.load_model_by_name("dtacs4bands")
model = model.to(device)
model.eval()

patch_size = 1024
margin = 128
step_size = patch_size - (2 * margin)

with rasterio.open(nir_path) as src_nir:
    total_height = src_nir.height
    total_width = src_nir.width
    meta = src_nir.meta.copy()

meta.update(dtype=rasterio.uint8, count=1, compress="lzw")

with rasterio.open(nir_path) as src_nir, \
     rasterio.open(red_path) as src_red, \
     rasterio.open(green_path) as src_green, \
     rasterio.open(output_path, "w", **meta) as dst_out:

    print(f"Generating LISS-IV enhanced cloud mask...")

    for y in tqdm(range(0, total_height, step_size), desc="Rows"):
        for x in range(0, total_width, step_size):

            read_window = Window(x - margin, y - margin, patch_size, patch_size)

            nir_block = src_nir.read(1, window=read_window, boundless=True, fill_value=0).astype(np.float32)
            red_block = src_red.read(1, window=read_window, boundless=True, fill_value=0).astype(np.float32)
            green_block = src_green.read(1, window=read_window, boundless=True, fill_value=0).astype(np.float32)

            blue_block = green_block.copy()

            stacked_block = np.stack([blue_block, green_block, red_block, nir_block], axis=0)
            stacked_block = stacked_block / 1023.0

            input_tensor = torch.from_numpy(stacked_block).unsqueeze(0).to(device)

            with torch.inference_mode():

                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
                    prediction = model(input_tensor)

            prediction_probs = prediction.cpu().squeeze().numpy()

            prediction_probs = prediction.cpu().squeeze().numpy()

            cloud_mask_block = (prediction_probs > 0.08).astype(np.uint8)

            epsilon = 1e-8

            ndvi = (nir_block - red_block) / (nir_block + red_block + epsilon)
            veg_mask = ndvi > 0.50

            soil_mask = red_block > (green_block * 1.4)

            dark_mask = (nir_block < 60) | (green_block < 60)

            cloud_mask_block[veg_mask] = 0
            cloud_mask_block[soil_mask] = 0
            cloud_mask_block[dark_mask] = 0

            bright_haze_mask = (green_block > 350) & (red_block > 350) & (nir_block > 350)

            cloud_mask_block = np.logical_or(cloud_mask_block, bright_haze_mask).astype(np.uint8)

            structuring_element = ndimage.generate_binary_structure(2, 2)

            cloud_mask_block = ndimage.binary_closing(
                cloud_mask_block,
                structure=structuring_element,
                iterations=2
            ).astype(np.uint8)

            cloud_mask_block = ndimage.binary_dilation(
                cloud_mask_block,
                structure=structuring_element,
                iterations=5
            ).astype(np.uint8)

            valid_center = cloud_mask_block[margin : patch_size - margin, margin : patch_size - margin]

            write_w = min(step_size, total_width - x)
            write_h = min(step_size, total_height - y)
            write_window = Window(x, y, write_w, write_h)

            dst_out.write(valid_center[:write_h, :write_w], 1, window=write_window)

print(f"\nFinal cleaned cloud mask saved to {output_path}")