import torch
import torch.nn as nn
import torch.nn.functional as F
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, use_norm=True, use_act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups, bias=not use_norm, padding_mode='reflect')
        self.norm = nn.GroupNorm(out_channels, out_channels) if use_norm else nn.Identity()
        self.act = nn.ReLU() if use_act else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class ComplexFiltering(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1_real = ConvLayer(in_channels, out_channels, 1, padding=0, use_norm=False, use_act=True)
        self.conv3_real = ConvLayer(out_channels, out_channels, 3, padding=1, use_norm=False, use_act=False)
        self.conv1_imag = ConvLayer(in_channels, out_channels, 1, padding=0, use_norm=False, use_act=True)
        self.conv3_imag = ConvLayer(out_channels, out_channels, 3, padding=1, use_norm=False, use_act=False)

    def forward(self, x):

        real_out = self.conv3_real(self.conv1_real(x.real))
        imag_out = self.conv3_imag(self.conv1_imag(x.imag))

        stacked_out = torch.stack([real_out.float(), imag_out.float()], dim=-1)
        return torch.view_as_complex(stacked_out.contiguous())

class MCL(nn.Module):
    def __init__(self, channels):
        super().__init__()
        c3 = channels // 3
        c_rem = channels - 2 * c3

        self.conv1_1 = ConvLayer(channels, c3, 1, padding=0)
        self.conv1_2 = ConvLayer(channels, c3, 1, padding=0)
        self.conv1_3 = ConvLayer(channels, c_rem, 1, padding=0)

        self.dwconv1 = ConvLayer(c3, c3, 3, padding=1, groups=c3)
        self.dwconv2 = ConvLayer(c3, c3, 3, padding=1, groups=c3)
        self.dwconv3 = ConvLayer(c_rem, c_rem, 3, padding=1, groups=c_rem)

        self.fuse = ConvLayer(channels, channels, 1, padding=0, use_act=False)

    def forward(self, x):
        x1 = self.conv1_1(x)
        x2 = self.conv1_2(x)
        x3 = self.conv1_3(x)

        y1 = self.dwconv1(x1)

        diff = x2.shape[1] - y1.shape[1]
        y1_p = F.pad(y1, (0, 0, 0, 0, 0, diff)) if diff > 0 else y1[:, :x2.shape[1]]
        y2 = self.dwconv2(x2 + y1_p)

        diff2 = x3.shape[1] - y2.shape[1]
        y2_p = F.pad(y2, (0, 0, 0, 0, 0, diff2)) if diff2 > 0 else y2[:, :x3.shape[1]]
        y3 = self.dwconv3(x3 + y2_p)

        return x + self.fuse(torch.cat([y1, y2, y3], dim=1))

class SLSA(nn.Module):
    def __init__(self, channels, window_size=8, num_heads=4):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.channels = channels

        self.pos_embed = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.norm = nn.GroupNorm(1, channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.v_conv1x1 = ConvLayer(channels, channels, 1, padding=0)
        self.v_conv3x3 = ConvLayer(channels, channels, 3, padding=1)

    def window_partition(self, x):
        B, C, H, W = x.shape
        ws = self.window_size
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(B, H // ws, ws, W // ws, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        return x.view(-1, ws * ws, C), H, W

    def window_reverse(self, windows, H, W, B):
        C = self.channels
        ws = self.window_size
        x = windows.view(B, H // ws, W // ws, ws, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H, W, C)
        return x.permute(0, 3, 1, 2).contiguous()

    def forward(self, x):
        B, C, H, W = x.shape
        x_pe = x + self.pos_embed(x)
        x_pe = self.norm(x_pe)

        windows, H, W = self.window_partition(x_pe)
        nW_B = windows.shape[0]
        head_dim = C // self.num_heads

        qkv = self.qkv(windows).reshape(nW_B, self.window_size**2, 3, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1) * head_dim ** -0.5).softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(nW_B, self.window_size**2, C)
        attn_out = self.window_reverse(out, H, W, B)

        v_spatial = self.window_reverse(v.transpose(1, 2).reshape(nW_B, self.window_size**2, C), H, W, B)
        v_out = self.v_conv3x3(self.v_conv1x1(v_spatial))

        return x + attn_out + v_out

class FGSA(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.filter_q = ComplexFiltering(channels, channels)
        self.filter_k = ComplexFiltering(channels, channels)

        self.v_conv1x1 = ConvLayer(channels, channels, 1, padding=0)
        self.v_conv3x3 = ConvLayer(channels, channels, 3, padding=1, use_act=False)
        self.ffn = nn.Sequential(
            ConvLayer(channels, channels * 4, 1, padding=0),
            ConvLayer(channels * 4, channels, 1, padding=0, use_act=False)
        )

    def forward(self, x):
        with torch.autocast(device_type='cuda', enabled=False):
            x_f = torch.fft.fft2(x.float(), norm='ortho')
            q_f = self.filter_q(x_f)
            k_f = self.filter_k(x_f)

            A = F.softmax(torch.fft.ifft2(q_f * k_f, norm='ortho').real, dim=1)

        v = self.v_conv3x3(self.v_conv1x1(x))

        return x + self.ffn(A * v)

class ReconBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.mcl = MCL(channels)
        self.slsa = SLSA(channels)
        self.fgsa = FGSA(channels)

    def forward(self, x):
        x = self.mcl(x)
        x = self.slsa(x)
        x = self.fgsa(x)
        return x

class FGCA_Aux(nn.Module):
    """
    4-way Frequency-Guided Cross Attention
    Takes f_opt, f_sar, f_temp, f_dem.
    """
    def __init__(self, channels):
        super().__init__()
        self.filter_opt = ComplexFiltering(channels, channels)
        self.filter_sar = ComplexFiltering(channels, channels)
        self.filter_temp = ComplexFiltering(channels, channels)
        self.filter_dem = ComplexFiltering(channels, channels)

        self.filter_freq = ComplexFiltering(channels * 4, channels)

        self.conv1x1_opt = nn.Conv2d(channels, channels, 1)
        self.conv1x1_sar = nn.Conv2d(channels, channels, 1)
        self.conv1x1_temp = nn.Conv2d(channels, channels, 1)
        self.conv1x1_dem = nn.Conv2d(channels, channels, 1)

        self.conv3x3_fuse = ConvLayer(channels * 4, channels, 3, padding=1, use_act=False)

    def forward(self, f_opt, f_sar, f_temp, f_dem):
        with torch.autocast(device_type='cuda', enabled=False):
            f_opt_f = torch.fft.fft2(f_opt.float(), norm='ortho')
            f_sar_f = torch.fft.fft2(f_sar.float(), norm='ortho')
            f_temp_f = torch.fft.fft2(f_temp.float(), norm='ortho')
            f_dem_f = torch.fft.fft2(f_dem.float(), norm='ortho')

            f_opt_f_filtered = self.filter_opt(f_opt_f)
            f_sar_f_filtered = self.filter_sar(f_sar_f)
            f_temp_f_filtered = self.filter_temp(f_temp_f)
            f_dem_f_filtered = self.filter_dem(f_dem_f)

            f_freq = self.filter_freq(torch.cat([
                f_opt_f_filtered, f_sar_f_filtered, f_temp_f_filtered, f_dem_f_filtered
            ], dim=1))

            cross_opt = f_freq * f_opt_f_filtered
            cross_sar = f_freq * f_sar_f_filtered
            cross_temp = f_freq * f_temp_f_filtered
            cross_dem = f_freq * f_dem_f_filtered

            opt_1x1 = torch.complex(self.conv1x1_opt(cross_opt.real), self.conv1x1_opt(cross_opt.imag))
            sar_1x1 = torch.complex(self.conv1x1_sar(cross_sar.real), self.conv1x1_sar(cross_sar.imag))
            temp_1x1 = torch.complex(self.conv1x1_temp(cross_temp.real), self.conv1x1_temp(cross_temp.imag))
            dem_1x1 = torch.complex(self.conv1x1_dem(cross_dem.real), self.conv1x1_dem(cross_dem.imag))

            f_opt_s = torch.fft.ifft2(opt_1x1, norm='ortho').real
            f_sar_s = torch.fft.ifft2(sar_1x1, norm='ortho').real
            f_temp_s = torch.fft.ifft2(temp_1x1, norm='ortho').real
            f_dem_s = torch.fft.ifft2(dem_1x1, norm='ortho').real

            attn_opt = F.softmax(f_opt_s, dim=1)
            attn_sar = F.softmax(f_sar_s, dim=1)
            attn_temp = F.softmax(f_temp_s, dim=1)
            attn_dem = F.softmax(f_dem_s, dim=1)

        return self.conv3x3_fuse(torch.cat([
            attn_opt * f_opt,
            attn_sar * f_sar,
            attn_temp * f_temp,
            attn_dem * f_dem
        ], dim=1))

class SLCA_Aux(nn.Module):
    """
    4-way Spatial-Level Cross Attention.
    Each query attends to a concatenated set of keys from all 4 modalities.
    """
    def __init__(self, channels, window_size=8, num_heads=4):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.channels = channels

        self.pos_embed_opt = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.pos_embed_sar = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.pos_embed_temp = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.pos_embed_dem = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)

        self.norm_opt = nn.GroupNorm(1, channels)
        self.norm_sar = nn.GroupNorm(1, channels)
        self.norm_temp = nn.GroupNorm(1, channels)
        self.norm_dem = nn.GroupNorm(1, channels)

        self.qkv_opt = nn.Linear(channels, channels * 3)
        self.qkv_sar = nn.Linear(channels, channels * 3)
        self.qkv_temp = nn.Linear(channels, channels * 3)
        self.qkv_dem = nn.Linear(channels, channels * 3)

        def make_ffn():
            return nn.Sequential(
                ConvLayer(channels, channels * 4, 1, padding=0),
                ConvLayer(channels * 4, channels, 1, padding=0, use_act=False)
            )

        self.ffn_opt = make_ffn()
        self.ffn_sar = make_ffn()
        self.ffn_temp = make_ffn()
        self.ffn_dem = make_ffn()

        self.fuse = ConvLayer(channels * 4, channels, 1, padding=0, use_act=False)

    def window_partition(self, x):
        B, C, H, W = x.shape
        ws = self.window_size
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(B, H // ws, ws, W // ws, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        return x.view(-1, ws * ws, C), H, W

    def window_reverse(self, windows, H, W, B):
        C = self.channels
        ws = self.window_size
        x = windows.view(B, H // ws, W // ws, ws, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H, W, C)
        return x.permute(0, 3, 1, 2).contiguous()

    def get_qkv(self, x_proj):

        x_reshaped = x_proj.reshape(-1, self.window_size**2, 3, self.num_heads, self.channels // self.num_heads).permute(2, 0, 3, 1, 4)
        return x_reshaped[0], x_reshaped[1], x_reshaped[2]

    def forward(self, f_opt, f_sar, f_temp, f_dem):
        B, C, H, W = f_opt.shape

        f_opt = self.norm_opt(f_opt + self.pos_embed_opt(f_opt))
        f_sar = self.norm_sar(f_sar + self.pos_embed_sar(f_sar))
        f_temp = self.norm_temp(f_temp + self.pos_embed_temp(f_temp))
        f_dem = self.norm_dem(f_dem + self.pos_embed_dem(f_dem))

        w_opt, H, W = self.window_partition(f_opt)
        w_sar, _, _ = self.window_partition(f_sar)
        w_temp, _, _ = self.window_partition(f_temp)
        w_dem, _, _ = self.window_partition(f_dem)

        scale = (C // self.num_heads) ** -0.5

        q_opt, k_opt, v_opt = self.get_qkv(self.qkv_opt(w_opt))
        q_sar, k_sar, v_sar = self.get_qkv(self.qkv_sar(w_sar))
        q_temp, k_temp, v_temp = self.get_qkv(self.qkv_temp(w_temp))
        q_dem, k_dem, v_dem = self.get_qkv(self.qkv_dem(w_dem))

        q_context_opt = (q_sar + q_temp + q_dem) / 3.0
        q_context_sar = (q_opt + q_temp + q_dem) / 3.0
        q_context_temp = (q_opt + q_sar + q_dem) / 3.0
        q_context_dem = (q_opt + q_sar + q_temp) / 3.0

        def compute_cross_attn(q_ctx, k, v):

            attn = (q_ctx @ k.transpose(-2, -1) * scale).softmax(dim=-1)

            ca = (attn @ v).transpose(1, 2).reshape(-1, self.window_size**2, C)

            return ca + v.transpose(1, 2).reshape(-1, self.window_size**2, C)

        ca_opt = compute_cross_attn(q_context_opt, k_opt, v_opt)
        ca_sar = compute_cross_attn(q_context_sar, k_sar, v_sar)
        ca_temp = compute_cross_attn(q_context_temp, k_temp, v_temp)
        ca_dem = compute_cross_attn(q_context_dem, k_dem, v_dem)

        out_opt = self.ffn_opt(self.window_reverse(ca_opt, H, W, B)) + f_opt
        out_sar = self.ffn_sar(self.window_reverse(ca_sar, H, W, B)) + f_sar
        out_temp = self.ffn_temp(self.window_reverse(ca_temp, H, W, B)) + f_temp
        out_dem = self.ffn_dem(self.window_reverse(ca_dem, H, W, B)) + f_dem

        return self.fuse(torch.cat([out_opt, out_sar, out_temp, out_dem], dim=1))

class SFFA_Aux(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fgca = FGCA_Aux(channels)
        self.slca = SLCA_Aux(channels)
        self.fuse = ConvLayer(channels * 2, channels, 3, padding=1, use_act=False)

    def forward(self, f_opt, f_sar, f_temp, f_dem):
        out_global = self.fgca(f_opt, f_sar, f_temp, f_dem)
        out_local  = self.slca(f_opt, f_sar, f_temp, f_dem)
        return self.fuse(torch.cat([out_global, out_local], dim=1))

class CMGR_Aux(nn.Module):
    """
    Cloud Mask Guided Rectification integrating DEM directly.
    """
    def __init__(self, channels):
        super().__init__()
        self.mask_proj = nn.Conv2d(1, channels, 3, padding=1)
        self.mcl = MCL(channels)

        self.conv = nn.Conv2d(channels * 4, channels, 3, padding=1)
        self.norm = nn.GroupNorm(1, channels)

    def forward(self, mask, f_opt, f_sar, f_dem, f_temp, f_fuse):

        m = self.mask_proj(mask)
        m = self.mcl(m)
        m = torch.sigmoid(m)

        concat_feat = torch.cat([f_opt * m, f_sar * m, f_dem*m, f_temp*m], dim=1)

        w = torch.sigmoid(self.norm(self.conv(concat_feat)))

        return w * f_fuse + f_fuse, w
