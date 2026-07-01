import os
import re
import argparse
import gc
from datetime import datetime, timedelta
from pathlib import Path

import ee
import geemap
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.warp import reproject, Resampling, transform_bounds
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt

from models.mrnet_lissiv import MRNet_LISSIV


MODEL_CONFIG = {
    "in_channels":      13,
    "out_channels_opt": 3,
    "out_channels_sar": 2,
    "feature_sizes":    256,
    "num_arb_layers":   16,
    "alpha":            0.1,
    "resolution":        256,
    "window_size":       8,
}

CLOUD_THRESHOLD = 0.10


def init_ee():
    PROJECT_ID = "gen-lang-client-0505852744"
    try:
        ee.Initialize(project=PROJECT_ID)
    except Exception as e:
        print("Earth Engine initialization failed. Please authenticate via `earthengine authenticate`.")
        raise e


def mask_s2_clouds(image):
    scl = image.select("SCL")
    mask = (
        scl.neq(1)
        .And(scl.neq(3))
        .And(scl.neq(8))
        .And(scl.neq(9))
        .And(scl.neq(10))
    )
    return image.updateMask(mask).divide(10000)


def download_and_align_aux(folder_path, wgs_bounds, start_date, end_date, target_date,
                            master_crs, master_transform, master_meta):
    """
    Downloads and reprojects the auxiliary inputs required by the model:
    SAR (VV, VH), temporal optical reference, and DEM.
    """
    downloads = {
        "aligned_sar.tif":      {"col": "COPERNICUS/S1_GRD",           "scale": 10, "band": ["VV", "VH"]},
        "aligned_temporal.tif": {"col": "COPERNICUS/S2_SR_HARMONIZED", "scale": 10, "band": ["B4", "B3", "B2"]},
        "aligned_dem.tif":      {"col": "COPERNICUS/DEM/GLO30",        "scale": 30, "band": ["DEM"]},
    }

    for aligned_name, info in downloads.items():
        aligned_path = folder_path / aligned_name
        if aligned_path.exists():
            continue

        init_ee()

        roi = ee.Geometry.BBox(wgs_bounds[0], wgs_bounds[1], wgs_bounds[2], wgs_bounds[3])
        target_ee_date = ee.Date(target_date.strftime("%Y-%m-%d"))

        print(f"  Fetching {aligned_name}...")
        raw_path = folder_path / ("raw_" + aligned_name)

        if info["col"] == "COPERNICUS/S2_SR_HARMONIZED":
            img = (ee.ImageCollection(info["col"]).filterBounds(roi).filterDate(start_date, end_date)
                   .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
                   .map(mask_s2_clouds).median().select(info["band"]).clip(roi))

        elif info["col"] == "COPERNICUS/S1_GRD":
            sar_col = (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(roi).filterDate(start_date, end_date)
                       .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
                       .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
                       .filter(ee.Filter.eq("instrumentMode", "IW")).select(["VV", "VH"]))
            sar_contains = sar_col.filter(ee.Filter.contains(".geo", roi))
            if sar_contains.size().getInfo() > 0:
                sar_col = sar_contains
            sar_col = sar_col.map(lambda image: image.set("timediff", image.date().difference(target_ee_date, "day").abs()))
            if sar_col.size().getInfo() == 0:
                print("  No SAR found!")
                return False
            img = ee.Image(sar_col.sort("timediff").first()).clip(roi)

        else:
            img = ee.ImageCollection(info["col"]).filterBounds(roi).select("DEM").mosaic()

        geemap.download_ee_image(img, str(raw_path), region=roi, scale=info["scale"],
                                  max_tile_size=16, max_tile_dim=1024, crs=master_crs.to_string())

        with rasterio.open(raw_path) as src_aux:
            kwargs = master_meta.copy()
            kwargs.update(dtype=src_aux.dtypes[0], count=src_aux.count, nodata=src_aux.nodata)
            with rasterio.open(aligned_path, "w", **kwargs) as dst:
                for b in range(1, src_aux.count + 1):
                    reproject(
                        source=rasterio.band(src_aux, b), destination=rasterio.band(dst, b),
                        src_transform=src_aux.transform, src_crs=src_aux.crs,
                        dst_transform=master_transform, dst_crs=master_crs,
                        src_nodata=src_aux.nodata, dst_nodata=src_aux.nodata, resampling=Resampling.bilinear
                    )
        raw_path.unlink()
        print(f"  Saved {aligned_path.name}")
    return True


def pad_tensor(t, expected_h, expected_w):
    _, _, h, w = t.shape
    pad_h = expected_h - h
    pad_w = expected_w - w
    if pad_h > 0 or pad_w > 0:
        t = F.pad(t, (0, pad_w, 0, pad_h), mode='reflect')
    return t


def n01(x, eps=1e-6):
    return (x - x.min()) / (x.max() - x.min() + eps)


def run_inference_on_folder(folder_path, weights_path, device='cuda'):
    print(f"\n{'='*40}")
    print(f"--- Processing {folder_path} ---")

    try:
        cloud_dir = next(folder_path.glob("*cloud*"))
        c2_path = next(cloud_dir.rglob("BAND2.tif"))
        c3_path = next(cloud_dir.rglob("BAND3.tif"))
        c4_path = next(cloud_dir.rglob("BAND4.tif"))
    except StopIteration:
        print("Missing required cloudy BAND files. Skipping.")
        return

    with rasterio.open(c2_path) as src_master:
        master_meta = src_master.meta.copy()
        master_transform = src_master.transform
        master_crs = src_master.crs
        wgs_bounds = transform_bounds(master_crs, "EPSG:4326", *src_master.bounds)
        H, W = src_master.height, src_master.width

    try:
        meta_file = next(cloud_dir.rglob("*META*.txt"))
        with open(meta_file, "r") as f:
            match = re.search(r"DateOfPass=\s*(\d{2}-\w{3}-\d{4})", f.read())
            if not match:
                raise ValueError("Date not found.")
            target_date = datetime.strptime(match.group(1), "%d-%b-%Y")
    except (StopIteration, ValueError):
        print("DateOfPass not found in metadata. Skipping.")
        return

    start_date = (target_date - timedelta(days=120)).strftime("%Y-%m-%d")
    end_date   = (target_date + timedelta(days=120)).strftime("%Y-%m-%d")

    success = download_and_align_aux(folder_path, wgs_bounds, start_date, end_date, target_date,
                                      master_crs, master_transform, master_meta)
    if not success:
        return

    print("  Initializing memory-mapped temporary buffers on disk...")
    sum_file        = folder_path / "temp_sum.dat"
    count_file      = folder_path / "temp_count.dat"
    mask_sum_file   = folder_path / "temp_mask_sum.dat"
    final_mask_file = folder_path / "temp_final_mask.dat"

    sum_mode   = 'r+' if sum_file.exists() else 'w+'
    count_mode = 'r+' if count_file.exists() else 'w+'
    output_sum   = np.memmap(sum_file,   dtype='float32', mode=sum_mode,   shape=(3, H, W))
    output_count = np.memmap(count_file, dtype='float32', mode=count_mode, shape=(1, H, W))

    if sum_mode == 'w+':
        output_sum[:] = 0.0
        output_count[:] = 0.0

    TILE_SIZE = MODEL_CONFIG["resolution"]
    STRIDE = 32
    BORDER = 32
    CORE_SIZE = TILE_SIZE - 2 * BORDER

    pad_h = (STRIDE - (H - TILE_SIZE) % STRIDE) % STRIDE
    pad_w = (STRIDE - (W - TILE_SIZE) % STRIDE) % STRIDE
    Hp = H + pad_h
    Wp = W + pad_w

    rows = list(range(0, Hp - TILE_SIZE + 1, STRIDE))
    cols = list(range(0, Wp - TILE_SIZE + 1, STRIDE))
    total_tiles = len(rows) * len(cols)

    src_c2  = rasterio.open(c2_path)
    src_c3  = rasterio.open(c3_path)
    src_c4  = rasterio.open(c4_path)
    src_sar = rasterio.open(folder_path / "aligned_sar.tif")
    src_tmp = rasterio.open(folder_path / "aligned_temporal.tif")
    src_dem = rasterio.open(folder_path / "aligned_dem.tif")

    is_cuda = (device.type if hasattr(device, 'type') else str(device)).startswith('cuda')

    def close_readers():
        src_c2.close(); src_c3.close(); src_c4.close()
        src_sar.close(); src_tmp.close(); src_dem.close()

    if H < TILE_SIZE or W < TILE_SIZE:
        print(f"  ERROR: Image ({H}x{W}) is smaller than TILE_SIZE ({TILE_SIZE}). Cannot proceed.")
        close_readers()
        return

    def read_tile(r, c):
        win = Window(c, r, TILE_SIZE, TILE_SIZE)
        c3_np = src_c3.read(1, window=win)
        c2_np = src_c2.read(1, window=win)
        c4_np = src_c4.read(1, window=win)
        sar_np = src_sar.read(window=win)
        tmp_np = src_tmp.read(window=win)
        dem_np = src_dem.read(1, window=win)
        return c3_np, c2_np, c4_np, sar_np, tmp_np, dem_np

    global_opt_scale = 1023.0
    chunk_size = 1000

    from utils.cloud_detection import Cloudsen12Detector

    if final_mask_file.exists():
        print(f"  [Phase 1/3] Found existing mask at {final_mask_file.name}. Skipping Mask Generation!")
        final_mask = np.memmap(final_mask_file, dtype='float32', mode='r', shape=(1, H, W))
        output_sum[:] = 0.0
        output_count[:] = 0.0
    else:
        print(f"  [Phase 1/3] Streaming {total_tiles} tiles for Cloud Mask Generation...")
        mask_sum = np.memmap(mask_sum_file, dtype='float32', mode='w+', shape=(1, H, W))
        mask_sum[:] = 0.0

        detector = Cloudsen12Detector(device=device)

        with torch.no_grad():
            done = 0
            for r in rows:
                for c in cols:
                    c3_np, c2_np, c4_np, _, _, _ = read_tile(r, c)
                    if c3_np.size == 0:
                        continue

                    cloudy_np = np.stack([c2_np, c3_np, c4_np], axis=0).astype(np.float32)
                    cloudy_t = torch.from_numpy(cloudy_np).unsqueeze(0)
                    cloudy_t = pad_tensor(cloudy_t, TILE_SIZE, TILE_SIZE).to(device)
                    cloudy_t = torch.clamp(cloudy_t / global_opt_scale, 0.0, 1.0)

                    tile_mask_d = detector.predict_binary(cloudy_t)
                    core_mask = tile_mask_d[:, :, BORDER:BORDER+CORE_SIZE, BORDER:BORDER+CORE_SIZE].cpu().numpy()

                    cr, cc = r + BORDER, c + BORDER
                    valid_h = min(CORE_SIZE, H - cr)
                    valid_w = min(CORE_SIZE, W - cc)

                    if valid_h > 0 and valid_w > 0 and cr < H and cc < W:
                        mask_sum[:, cr:cr+valid_h, cc:cc+valid_w] += core_mask[0, :, :valid_h, :valid_w]
                        output_count[:, cr:cr+valid_h, cc:cc+valid_w] += 1.0

                    del tile_mask_d, cloudy_t
                    done += 1
                    if done % 50 == 0:
                        print(f"    Mask tiles: {done}/{total_tiles}")

        del detector
        if is_cuda:
            torch.cuda.empty_cache()
        gc.collect()

        print("  Processing final mask directly on disk...")
        final_mask = np.memmap(final_mask_file, dtype='float32', mode='w+', shape=(1, H, W))
        for y in range(0, H, chunk_size):
            for x in range(0, W, chunk_size):
                y2 = min(H, y + chunk_size)
                x2 = min(W, x + chunk_size)
                c_chunk = np.clip(np.array(output_count[:, y:y2, x:x2]), 1.0, None)
                m_chunk = np.array(mask_sum[:, y:y2, x:x2]) / c_chunk
                final_mask[:, y:y2, x:x2] = (m_chunk > CLOUD_THRESHOLD).astype(np.float32)

        output_count[:] = 0.0

    print(f"  [Phase 2/3] Loading MRNet_LISSIV model...")
    model = MRNet_LISSIV(
        in_channels   = MODEL_CONFIG["in_channels"],
        out_opt       = MODEL_CONFIG["out_channels_opt"],
        out_sar       = MODEL_CONFIG["out_channels_sar"],
        alpha         = MODEL_CONFIG["alpha"],
        num_layers    = MODEL_CONFIG["num_arb_layers"],
        feature_sizes = MODEL_CONFIG["feature_sizes"],
        resolution    = MODEL_CONFIG["resolution"],
        window_size   = MODEL_CONFIG["window_size"],
    ).to(device)

    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = None
    for key in ("state_dict", "model_state_dict", "model", "model_sd"):
        if isinstance(checkpoint, dict) and key in checkpoint:
            state_dict = checkpoint[key]
            break
    if state_dict is None:
        state_dict = checkpoint if not isinstance(checkpoint, dict) else checkpoint

    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[10:] if k.startswith('_orig_mod.') else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model.eval()

    window_1d = np.hamming(CORE_SIZE)
    window_2d = np.outer(window_1d, window_1d).astype(np.float32)

    print(f"  [Phase 3/3] Streaming {total_tiles} tiles for MRNet_LISSIV Inference...")
    with torch.no_grad():
        done = 0
        for r in rows:
            for c in cols:
                c3_np, c2_np, c4_np, sar_np, tmp_np, dem_np = read_tile(r, c)
                if c3_np.size == 0:
                    continue

                cloudy_np = np.stack([c2_np, c3_np, c4_np], axis=0).astype(np.float32)
                cloudy_n  = n01(np.clip(cloudy_np / global_opt_scale, 0.0, 1.0))

                sar_n = n01(sar_np.astype(np.float32))

                tmp_n = n01(tmp_np.astype(np.float32))

                dem_n = n01(dem_np.astype(np.float32))[np.newaxis]

                r2_mask = min(H, r + TILE_SIZE)
                c2_mask = min(W, c + TILE_SIZE)
                mask_np = np.array(final_mask[:, r:r2_mask, c:c2_mask]).astype(np.float32)

                dist_np = n01(distance_transform_edt(1 - mask_np[0]).astype(np.float32))[np.newaxis]

                ndvi_np = n01((tmp_n[2] - tmp_n[0]) / (tmp_n[2] + tmp_n[0] + 1e-6))[np.newaxis]

                inp_np = np.concatenate([
                    cloudy_n, sar_n, tmp_n, dem_n, mask_np, dist_np, ndvi_np
                ], axis=0).astype(np.float32)

                inp_t = pad_tensor(torch.from_numpy(inp_np).unsqueeze(0), TILE_SIZE, TILE_SIZE).to(device)

                pred_opt, pred_sar, feat = model(inp_t)
                pred_opt = torch.nan_to_num(pred_opt, nan=0.0, posinf=1.0, neginf=0.0)
                pred_opt = torch.clamp(pred_opt, 0.0, 1.0)

                core_pred = pred_opt[:, :, BORDER:BORDER+CORE_SIZE, BORDER:BORDER+CORE_SIZE].float().cpu().numpy()

                cr, cc = r + BORDER, c + BORDER
                valid_h = min(CORE_SIZE, H - cr)
                valid_w = min(CORE_SIZE, W - cc)

                if valid_h > 0 and valid_w > 0 and cr < H and cc < W:
                    w_tile = window_2d[:valid_h, :valid_w]
                    output_sum[:, cr:cr+valid_h, cc:cc+valid_w]   += core_pred[0, :, :valid_h, :valid_w] * w_tile
                    output_count[:, cr:cr+valid_h, cc:cc+valid_w] += w_tile

                del inp_t, pred_opt, pred_sar, feat
                done += 1
                if done % 50 == 0:
                    print(f"    Inference tiles: {done}/{total_tiles}")
                    if is_cuda:
                        torch.cuda.empty_cache()

    del model
    if is_cuda:
        torch.cuda.empty_cache()
    gc.collect()

    print("  Saving Output to GeoTIFF...")
    out_tif = folder_path / "mrnet_predicted_clear.tif"
    prof = master_meta.copy()
    prof.update(count=3, photometric='RGB', interleave='pixel')

    with rasterio.open(out_tif, 'w', **prof) as dst:
        for y in range(0, H, chunk_size):
            for x in range(0, W, chunk_size):
                y2 = min(H, y + chunk_size)
                x2 = min(W, x + chunk_size)

                c_chunk = np.clip(np.array(output_count[:, y:y2, x:x2]), 1.0, None)
                pred_chunk = np.array(output_sum[:, y:y2, x:x2]) / c_chunk

                pred_chunk = np.clip(pred_chunk, 0.0, 1.0)
                pred_chunk = (pred_chunk * global_opt_scale).astype(master_meta['dtype'])

                win = Window(x, y, x2 - x, y2 - y)
                dst.write(pred_chunk, window=win)

    print("  Creating false-color visualization...")
    decimate_factor = max(1, H // 1000)
    vis_h = max(1, H // decimate_factor)
    vis_w = max(1, W // decimate_factor)

    vis_cloudy_b3 = src_c3.read(1, out_shape=(vis_h, vis_w), resampling=Resampling.average)
    vis_cloudy_b2 = src_c2.read(1, out_shape=(vis_h, vis_w), resampling=Resampling.average)
    vis_cloudy_b4 = src_c4.read(1, out_shape=(vis_h, vis_w), resampling=Resampling.average)
    vis_cloudy_np = np.stack([vis_cloudy_b3, vis_cloudy_b2, vis_cloudy_b4], axis=0)

    vis_pred_np = np.array(output_sum[:, ::decimate_factor, ::decimate_factor]) / \
                  np.clip(np.array(output_count[:, ::decimate_factor, ::decimate_factor]), 1.0, None)
    vis_pred_np = vis_pred_np * global_opt_scale

    vis_mask_np = np.array(final_mask[0, ::decimate_factor, ::decimate_factor])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    def normalize_for_plot(img):
        img = img.transpose(1, 2, 0)
        fcc = np.stack([img[:, :, 2], img[:, :, 0], img[:, :, 1]], axis=-1)
        p2, p98 = np.percentile(fcc, (2, 98))
        return np.clip((fcc - p2) / (p98 - p2 + 1e-8), 0, 1)

    axes[0].imshow(normalize_for_plot(vis_cloudy_np))
    axes[0].set_title("Original Cloudy (False Color)")
    axes[0].axis("off")

    axes[1].imshow(vis_mask_np, cmap="gray")
    axes[1].set_title("Generated Cloud Mask")
    axes[1].axis("off")

    axes[2].imshow(normalize_for_plot(vis_pred_np))
    axes[2].set_title("MRNet Predicted Clear (False Color)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(folder_path / "mrnet_visualization.png", dpi=150)
    plt.close()

    output_sum.flush()
    output_count.flush()
    close_readers()

    print("  Cleaning up temporary files...")
    del output_sum, output_count, final_mask
    gc.collect()
    if sum_file.exists(): sum_file.unlink()
    if count_file.exists(): count_file.unlink()
    if mask_sum_file.exists(): mask_sum_file.unlink()
    if final_mask_file.exists(): final_mask_file.unlink()

    print(f"--- Finished {folder_path} ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MRNet_LISSIV Inference Pipeline")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to data directory containing folders 1, 2, etc.")
    parser.add_argument("--weights", type=str, default="mrnet_weights.pth", help="Path to model weights")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")

    args = parser.parse_args()

    base_dir = Path(args.data_dir)
    if not base_dir.exists():
        print(f"Data directory {base_dir} not found.")
        exit(1)

    if list(base_dir.glob("*cloud*")):
        run_inference_on_folder(base_dir, args.weights, device=torch.device(args.device))
    else:
        found = False
        for item in sorted(base_dir.iterdir()):
            if item.is_dir() and list(item.glob("*cloud*")):
                run_inference_on_folder(item, args.weights, device=torch.device(args.device))
                found = True

        if not found:
            print(f"No '*cloud*' directories found in {base_dir} or its immediate subdirectories.")
