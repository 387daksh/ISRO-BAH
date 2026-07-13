import torch
import torch.nn as nn
from .modules_aux import SFFA_Aux, CMGR_Aux, MCL, ReconBlock, ConvLayer

class DownsampleAux(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False, padding_mode='reflect'),
            nn.GroupNorm(out_channels, out_channels),
            nn.GELU()
        )

    def forward(self, x):
        return self.conv(x)

class UpsampleAux(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, padding_mode='reflect'),
            nn.GroupNorm(out_channels, out_channels),
            nn.GELU()
        )

    def forward(self, x):
        return self.conv(x)

class PSF_CR_Aux(nn.Module):
    """
    4-way Auxiliary PSF-CR Model.
    Takes Cloudy Optical (3), SAR (2), DEM (1), Temporal (4).
    """
    def __init__(self, in_channels_opt=3, in_channels_sar=2, in_channels_dem=1, in_channels_temp=4, base_channels=64):
        super().__init__()

        self.init_opt = nn.Conv2d(in_channels_opt, base_channels, 3, padding=1, padding_mode='reflect')
        self.init_sar = nn.Conv2d(in_channels_sar, base_channels, 3, padding=1, padding_mode='reflect')
        self.init_dem = nn.Conv2d(in_channels_dem, base_channels, 3, padding=1, padding_mode='reflect')
        self.init_temp = nn.Conv2d(in_channels_temp, base_channels, 3, padding=1, padding_mode='reflect')

        self.mcl_opt = MCL(base_channels)
        self.mcl_sar = MCL(base_channels)
        self.mcl_dem = MCL(base_channels)
        self.mcl_temp = MCL(base_channels)

        self.sffa = SFFA_Aux(base_channels)
        self.cmgr = CMGR_Aux(base_channels)

        self.stage1 = ReconBlock(base_channels)
        self.down1 = DownsampleAux(base_channels, base_channels * 2)

        self.stage2 = ReconBlock(base_channels * 2)
        self.down2 = DownsampleAux(base_channels * 2, base_channels * 4)

        self.stage3 = ReconBlock(base_channels * 4)

        self.up1 = UpsampleAux(base_channels * 4, base_channels * 2)
        self.reduce1 = nn.Conv2d(base_channels * 4, base_channels * 2, 1)
        self.stage4 = ReconBlock(base_channels * 2)

        self.up2 = UpsampleAux(base_channels * 2, base_channels)
        self.reduce2 = nn.Conv2d(base_channels * 2, base_channels, 1)
        self.stage5 = ReconBlock(base_channels)

        self.pre_final = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 1, padding=0, padding_mode='reflect'),
            nn.GroupNorm(base_channels, base_channels),
            nn.ReLU()
        )

        self.final_conv = nn.Conv2d(base_channels, 3, 3, padding=1, padding_mode='reflect')

    def forward(self, opt, sar, dem, temporal, mask):

        f_opt = self.mcl_opt(self.init_opt(opt))
        f_sar = self.mcl_sar(self.init_sar(sar))
        f_dem = self.mcl_dem(self.init_dem(dem))
        f_temp = self.mcl_temp(self.init_temp(temporal))

        f_fuse = self.sffa(f_opt, f_sar, f_temp, f_dem)

        f_rectified, w_matrix = self.cmgr(mask, f_opt, f_sar, f_dem, f_temp, f_fuse)

        x1 = self.stage1(f_rectified)
        x1_down = self.down1(x1)

        x2 = self.stage2(x1_down)
        x2_down = self.down2(x2)

        x3 = self.stage3(x2_down)

        x4_up = self.up1(x3)
        x4_concat = torch.cat([x4_up, x2], dim=1)
        x4_reduced = self.reduce1(x4_concat)
        x4 = self.stage4(x4_reduced)

        x5_up = self.up2(x4)
        x5_concat = torch.cat([x5_up, x1], dim=1)
        x5_reduced = self.reduce2(x5_concat)
        x5 = self.stage5(x5_reduced)

        x5_safe = self.pre_final(x5)

        out = self.final_conv(x5_safe)

        return out, w_matrix
