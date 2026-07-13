import os
import math
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from collections import defaultdict
import random
from models.psf_cr_aux import PSF_CR_Aux
from utils.dataset import SatDataset
from utils.losses import Phase2Loss
from utils.metrics import compute_all_metrics
from tqdm import tqdm

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay  = decay
        self.shadow = {name: p.data.clone().float()
                       for name, p in model.named_parameters() if p.requires_grad}

    def update(self, model):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    self.shadow[name] = (self.decay * self.shadow[name]
                                         + (1.0 - self.decay) * param.data.float())

    def apply(self, model):
        self._backup = {name: p.data.clone()
                        for name, p in model.named_parameters() if p.requires_grad}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                param.data.copy_(self.shadow[name].to(param.dtype))

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self._backup:
                param.data.copy_(self._backup[name])
        self._backup = {}

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, sd):
        self.shadow = sd

def apply_aug(opt_c, opt_gt, sar, dem, temporal, mask):
    if torch.rand(1).item() > 0.5:
        opt_c  = torch.flip(opt_c,  dims=[-1])
        opt_gt = torch.flip(opt_gt, dims=[-1])
        sar    = torch.flip(sar,    dims=[-1])
        dem    = torch.flip(dem,    dims=[-1])
        temporal = torch.flip(temporal, dims=[-1])
        mask   = torch.flip(mask,   dims=[-1])
    if torch.rand(1).item() > 0.5:
        opt_c  = torch.flip(opt_c,  dims=[-2])
        opt_gt = torch.flip(opt_gt, dims=[-2])
        sar    = torch.flip(sar,    dims=[-2])
        dem    = torch.flip(dem,    dims=[-2])
        temporal = torch.flip(temporal, dims=[-2])
        mask   = torch.flip(mask,   dims=[-2])
    k = torch.randint(0, 4, (1,)).item()
    if k > 0:
        opt_c  = torch.rot90(opt_c,  k, dims=[-2, -1])
        opt_gt = torch.rot90(opt_gt, k, dims=[-2, -1])
        sar    = torch.rot90(sar,    k, dims=[-2, -1])
        dem    = torch.rot90(dem,    k, dims=[-2, -1])
        temporal = torch.rot90(temporal, k, dims=[-2, -1])
        mask   = torch.rot90(mask,   k, dims=[-2, -1])

    return opt_c, opt_gt, sar, dem, temporal, mask

def save_training_visualization(model, loader, epoch, device, out_dir, num_samples=5):
    """Save a quick visual check during training using val patches.
    Saves a side-by-side PNG: [Cloudy FCC | GT FCC | Pred FCC | W_Matrix | Error Map]
    """
    import torchvision
    os.makedirs(out_dir, exist_ok=True)
    samples_saved = 0
    with torch.no_grad():
        for batch in loader:
            opt_c    = batch['cloudy'].to(device)
            sar      = batch['sar'].to(device)
            dem      = batch['dem'].to(device)
            temporal = batch['temporal'].to(device)
            opt_gt   = batch['clear'].to(device)
            mask     = batch['mask'].to(device)

            pred, w_matrix = model(opt_c, sar, dem, temporal, mask)
            pred = torch.clamp(pred, 0.0, 1.0)

            if w_matrix.size(1) > 1:
                w_vis = w_matrix.mean(dim=1, keepdim=True)
            else:
                w_vis = w_matrix
            w_vis = (w_vis - w_vis.min()) / (w_vis.max() - w_vis.min() + 1e-8)
            w_vis_3ch = w_vis.expand(-1, 3, -1, -1)

            err = torch.abs(pred - opt_gt).mean(dim=1, keepdim=True)
            err = (err - err.min()) / (err.max() - err.min() + 1e-8)
            err_3ch = err.expand(-1, 3, -1, -1)

            for i in range(opt_c.size(0)):
                if samples_saved >= num_samples:
                    return

                def to_fcc(img):
                    return torch.stack([img[2], img[1], img[0]], dim=0)

                row = torch.stack([
                    to_fcc(opt_c[i]),
                    to_fcc(opt_gt[i]),
                    to_fcc(pred[i]),
                    w_vis_3ch[i],
                    err_3ch[i]
                ])
                img_path = os.path.join(out_dir, f"epoch_{epoch:03d}_sample_{samples_saved:02d}.png")
                torchvision.utils.save_image(row, img_path, nrow=5, padding=2, pad_value=1.0)
                samples_saved += 1

def cosine_warmup_lr(optimizer, epoch, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg['lr'] = lr
        return lr

def mixup(opt_c, opt_gt, sar, dem, temporal, mask, alpha=0.2):
    if alpha <= 0:
        return opt_c, opt_gt, sar, dem, temporal, mask
    lam = float(torch.distributions.Beta(
        torch.tensor(alpha), torch.tensor(alpha)).sample())
    B   = opt_c.size(0)
    idx = torch.randperm(B, device=opt_c.device)
    return (lam * opt_c  + (1 - lam) * opt_c[idx],
            lam * opt_gt + (1 - lam) * opt_gt[idx],
            lam * sar    + (1 - lam) * sar[idx],
            lam * dem    + (1 - lam) * dem[idx],
            lam * temporal + (1 - lam) * temporal[idx],
            lam * mask   + (1 - lam) * mask[idx])

def train_approach2(
    data_root='/media/aaryaman/New Volume/training_data',
    epochs=50,
    batch_size=32,
    accumulation_steps=2,
    lr=1e-4,
    min_lr=1e-6,
    warmup_epochs=5,
    save_path='approach2_aux_weights.pth',
    load_path=None,
    patience=10,
    val_split=0.1,
    test_split=0.1,
    resume=True,
    viz_freq=5
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Approach 2 | 4-Way Auxiliary PSF-CR | Device: {device}")

    dataset = SatDataset(data_dir=data_root, require_temporal=True)

    tile_dict = defaultdict(list)
    for idx, fname in enumerate(dataset.filenames):
        tile_id = fname.split('_')[0] if '_' in fname else fname
        tile_dict[tile_id].append(idx)

    unique_tiles = sorted(list(tile_dict.keys()))

    random.seed(42)
    random.shuffle(unique_tiles)

    val_tiles_count = max(1, int(len(unique_tiles) * val_split))
    test_tiles_count = max(1, int(len(unique_tiles) * test_split)) if test_split > 0 else 0
    train_tiles_count = len(unique_tiles) - val_tiles_count - test_tiles_count

    train_tiles = unique_tiles[:train_tiles_count]
    val_tiles = unique_tiles[train_tiles_count : train_tiles_count + val_tiles_count]
    test_tiles = unique_tiles[train_tiles_count + val_tiles_count :]

    train_idx = [idx for tile in train_tiles for idx in tile_dict[tile]]
    val_idx = [idx for tile in val_tiles for idx in tile_dict[tile]]
    test_idx = [idx for tile in test_tiles for idx in tile_dict[tile]]

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    print(f"Dataset Split (Grouped by Tile): Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                              num_workers=8, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False,
                              num_workers=8, pin_memory=True)

    model = PSF_CR_Aux().to(device)

    try:
        if torch.__version__ >= "2.0.0":
            import warnings
            warnings.filterwarnings("ignore", category=UserWarning, module="torch._inductor")
            model = torch.compile(model)
            print("Successfully applied torch.compile().")
    except Exception as e:
        print(f"Skipping torch.compile(): {e}")

    criterion = Phase2Loss(lambda1=1.0, lambda2=0.1, lambda3=0.5).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-4)

    ema = EMA(model, decay=0.999)

    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

    scaler = torch.amp.GradScaler('cuda', enabled=False)

    early_stop_counter = 0

    if load_path is None:
        load_path = save_path

    start_epoch = 0
    best_psnr = 0.0

    if resume and os.path.exists(load_path):
        print(f"Resuming from {load_path}...")
        ckpt = torch.load(load_path, map_location=device)
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            if 'best_psnr' in ckpt:
                best_psnr = ckpt['best_psnr']

            if 'ema_state_dict' in ckpt:
                ema.load_state_dict(ckpt['ema_state_dict'])
                print(f"Resumed epoch {start_epoch}, Best PSNR: {best_psnr:.2f} (EMA loaded)")
            else:
                ema = EMA(model, decay=0.999)
                print(f"Resumed epoch {start_epoch}, Best PSNR: {best_psnr:.2f} (EMA re-initialized from loaded weights)")
        else:
            model.load_state_dict(ckpt)
            print("Loaded raw weights.")

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        batches = 0

        current_lr = cosine_warmup_lr(optimizer, epoch, warmup_epochs, epochs, lr, min_lr)

        optimizer.zero_grad()
        train_pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}] LR={current_lr:.1e} Train")

        for i, batch in enumerate(train_pbar):
            opt_c = batch['cloudy'].to(device)
            sar = batch['sar'].to(device)
            dem = batch['dem'].to(device)
            temporal = batch['temporal'].to(device)
            opt_gt = batch['clear'].to(device)
            mask = batch['mask'].to(device)

            dem_drop_mask = (torch.rand(dem.size(0), 1, 1, 1, device=device) > 0.35).float()
            dem = dem * dem_drop_mask

            temp_drop_mask = (torch.rand(temporal.size(0), 1, 1, 1, device=device) > 0.40).float()
            temporal = temporal * temp_drop_mask

            opt_c, opt_gt, sar, dem, temporal, mask = apply_aug(opt_c, opt_gt, sar, dem, temporal, mask)

            with torch.autocast(device_type='cuda', dtype=amp_dtype):
                pred, w_matrix = model(opt_c, sar, dem, temporal, mask)
                pred_c = pred
                gt_c = opt_gt

                w_c = w_matrix.detach()

                loss = criterion(pred_c, gt_c, w_c)
                loss = loss / accumulation_steps

            loss.backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                optimizer.zero_grad()
                if torch.isfinite(gnorm):
                    ema.update(model)
                else:
                    print(f"\n[Warning] NaN gradient at Epoch {epoch+1}, Step {i+1}. Skipping EMA update.")

            running_loss += loss.item() * accumulation_steps
            batches += 1
            train_pbar.set_postfix({'loss': f"{running_loss / batches:.4f}"})

        avg_loss = running_loss / batches

        ema.apply(model)
        model.eval()
        val_psnr = val_sam = val_ssim = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch [{epoch+1}/{epochs}] Val"):
                opt_c = batch['cloudy'].to(device)
                sar = batch['sar'].to(device)
                dem = batch['dem'].to(device)
                temporal = batch['temporal'].to(device)
                opt_gt = batch['clear'].to(device)
                mask = batch['mask'].to(device)

                pred, _ = model(opt_c, sar, dem, temporal, mask)

                pred_c = torch.clamp(pred, 0.0, 1.0)
                gt_c = opt_gt

                metrics = compute_all_metrics(pred_c, gt_c)
                val_psnr += metrics['psnr']
                val_sam += metrics['sam']
                val_ssim += metrics['ssim']
                val_batches += 1

        ema.restore(model)

        avg_val_psnr = val_psnr / val_batches
        avg_val_sam = val_sam / val_batches
        avg_val_ssim = val_ssim / val_batches

        print(
            f"Epoch [{epoch+1}/{epochs}] | LR: {current_lr:.2e} | Loss: {avg_loss:.4f} | "
            f"Val PSNR: {avg_val_psnr:.2f} | Val SAM: {avg_val_sam:.4f} | Val SSIM: {avg_val_ssim:.4f}"
        )

        if avg_val_psnr > best_psnr:
            best_psnr = avg_val_psnr
            ema.apply(model)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ema_state_dict': ema.state_dict(),
                'best_psnr': best_psnr
            }, save_path)
            ema.restore(model)
            print(f"  -> Best EMA model saved! PSNR: {best_psnr:.2f}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            current_patience = patience if epoch < 50 else max(5, patience // 2)
            print(f"  -> No improvement. Counter: {early_stop_counter}/{current_patience}")
            if early_stop_counter >= current_patience:
                print("Early stopping triggered!")
                break

        if (epoch + 1) % viz_freq == 0:
            periodic = save_path.replace('.pth', f'_epoch_{epoch+1}.pth')
            ema.apply(model)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'ema_state_dict': ema.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_psnr': best_psnr
            }, periodic)
            ema.restore(model)
            print(f"  -> Periodic checkpoint: {periodic}")

            vis_dir = os.path.join(os.path.dirname(os.path.abspath(save_path)), 'training_vis')
            ema.apply(model)
            model.eval()
            save_training_visualization(model, val_loader, epoch + 1, device, vis_dir, num_samples=5)
            ema.restore(model)
            model.train()

    print("Training Approach 2 Complete.")

    print("\n--- Final Test Set Evaluation ---")
    ckpt = torch.load(save_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    model.eval()
    test_psnr = test_sam = test_ssim = 0.0
    test_batches = 0

    with torch.no_grad():
        for batch in test_loader:
            opt_c = batch['cloudy'].to(device)
            sar = batch['sar'].to(device)
            dem = batch['dem'].to(device)
            temporal = batch['temporal'].to(device)
            opt_gt = batch['clear'].to(device)
            mask = batch['mask'].to(device)

            pred, _ = model(opt_c, sar, dem, temporal, mask)

            pred_c = pred
            gt_c = opt_gt

            pred_c = torch.clamp(pred_c, 0.0, 1.0)

            metrics = compute_all_metrics(pred_c, gt_c)
            test_psnr += metrics['psnr']
            test_sam += metrics['sam']
            test_ssim += metrics['ssim']
            test_batches += 1

    if test_batches > 0:
        print(
            f"Final Test PSNR: {test_psnr/test_batches:.2f} | "
            f"Test SAM: {test_sam/test_batches:.4f} | "
            f"Test SSIM: {test_ssim/test_batches:.4f}"
        )

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Train Approach 2 (Auxiliary PSF-CR)")
    parser.add_argument('--data_root', type=str, default='/media/aaryaman/New Volume/training_data')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--accumulation_steps', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--save_path', type=str, default='approach2_aux_weights.pth')
    parser.add_argument('--load_path', type=str, default=None, help="Specific checkpoint to load")
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--test_split', type=float, default=0.1)
    parser.add_argument('--no_resume', action='store_true', help="Disable resuming from checkpoint")
    parser.add_argument('--viz_freq', type=int, default=5, help="Frequency (in epochs) to save visualizations and periodic checkpoints")
    args = parser.parse_args()

    train_approach2(
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        accumulation_steps=args.accumulation_steps,
        lr=args.lr,
        min_lr=args.min_lr,
        warmup_epochs=args.warmup_epochs,
        save_path=args.save_path,
        load_path=args.load_path,
        patience=args.patience,
        val_split=args.val_split,
        test_split=args.test_split,
        resume=not args.no_resume,
        viz_freq=args.viz_freq
    )
