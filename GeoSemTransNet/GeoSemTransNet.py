import torch
import torch.nn as nn
import torch.nn.functional as F

class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc):
        super().__init__()
        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)

        nhidden = 128
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)

    def forward(self, x, condition_map):
        normalized = self.param_free_norm(x)
        if condition_map.size()[2:] != x.size()[2:]:
            condition_map = F.interpolate(condition_map, size=x.size()[2:], mode='nearest')

        actv = self.mlp_shared(condition_map)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        return normalized * (1 + gamma) + beta


class SPADEResBlock(nn.Module):
    def __init__(self, fin, fout, label_nc):
        super().__init__()
        self.learned_shortcut = (fin != fout)
        self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False) if self.learned_shortcut else None

        self.conv_0 = nn.Conv2d(fin, fout, kernel_size=3, padding=1, bias=False)
        self.spade_0 = SPADE(fin, label_nc)

        self.conv_1 = nn.Conv2d(fout, fout, kernel_size=3, padding=1, bias=False)
        self.spade_1 = SPADE(fout, label_nc)

    def forward(self, x, condition_map):
        x_s = self.conv_s(x) if self.learned_shortcut else x

        dx = self.spade_0(x, condition_map)
        dx = F.relu(dx)
        dx = self.conv_0(dx)

        dx = self.spade_1(dx, condition_map)
        dx = F.relu(dx)
        dx = self.conv_1(dx)

        return x_s + dx


class GeoSemTransNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(5, 64, 7, padding=3), nn.InstanceNorm2d(64), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2),  # 112x112
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.InstanceNorm2d(256), nn.LeakyReLU(0.2),  # 56x56
            nn.Conv2d(256, 512, 3, stride=2, padding=1), nn.InstanceNorm2d(512), nn.LeakyReLU(0.2),  # 28x28
            nn.Conv2d(512, 512, 3, stride=2, padding=1), nn.InstanceNorm2d(512), nn.LeakyReLU(0.2),  # 14x14
        )

        self.dino_proj = nn.Sequential(
            nn.Conv2d(384, 512, 1),
            nn.GroupNorm(32, 512),
            nn.GELU()
        )

        label_nc = 5

        self.head_0 = SPADEResBlock(512, 512, label_nc)

        self.up_1 = nn.Upsample(scale_factor=2)
        self.head_1 = SPADEResBlock(512, 512, label_nc)

        self.up_2 = nn.Upsample(scale_factor=2)
        self.head_2 = SPADEResBlock(512, 256, label_nc)

        self.up_3 = nn.Upsample(scale_factor=2)
        self.head_3 = SPADEResBlock(256, 128, label_nc)

        self.up_4 = nn.Upsample(scale_factor=2)
        self.head_4 = SPADEResBlock(128, 64, label_nc)

        self.final_conv = nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, src_img, mask, normals, alpha, fg, dino_feat):
        enc_input = torch.cat([src_img, fg, alpha], dim=1)
        feat_enc = self.encoder(enc_input)

        if dino_feat.dim() == 3:
            B, N, C = dino_feat.shape
            size = int(N ** 0.5)
            dino_feat = dino_feat.permute(0, 2, 1).view(B, C, size, size)

        feat_dino = self.dino_proj(dino_feat)  #

        x = feat_enc + feat_dino

        condition = torch.cat([normals, mask, fg], dim=1)  #

        x = self.head_0(x, condition)

        x = self.up_1(x)
        x = self.head_1(x, condition)

        x = self.up_2(x)
        x = self.head_2(x, condition)

        x = self.up_3(x)
        x = self.head_3(x, condition)

        x = self.up_4(x)
        x = self.head_4(x, condition)

        raw_out = torch.tanh(self.final_conv(x))  # [-1, 1] 范围
        final_out = ((1 + raw_out) / 2) * mask + src_img * (1 - mask)

        return final_out



