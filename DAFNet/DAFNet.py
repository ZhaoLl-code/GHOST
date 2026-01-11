import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class MMSA_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: 224x224
        self.stem = ConvBlock(6, 64, stride=2)

        self.layer1 = nn.Sequential(
            ConvBlock(64, 128, stride=2),
            ConvBlock(128, 128)
        )
        self.layer2 = nn.Sequential(
            ConvBlock(128, 256, stride=2),
            ConvBlock(256, 256)
        )
        self.layer3 = nn.Sequential(
            ConvBlock(256, 384, stride=2),
            ConvBlock(384, 384)
        )

    def forward(self, x):
        # x:
        c1 = self.stem(x)  # 112
        c2 = self.layer1(c1)  # 56
        c3 = self.layer2(c2)  # 28
        c4 = self.layer3(c3)  # 14
        return [c1, c2, c3, c4]

class FusionDecoderBlock(nn.Module):
    def __init__(self, in_channels_high, in_channels_skip, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.fusion_conv = ConvBlock(in_channels_high + in_channels_skip, out_channels)
        self.attention = SEBlock(out_channels)

    def forward(self, x_high, x_skip):
        x_up = self.upsample(x_high)
        if x_up.shape != x_skip.shape:
            x_up = F.interpolate(x_up, size=x_skip.shape[2:], mode='bilinear', align_corners=True)
        x_cat = torch.cat([x_up, x_skip], dim=1)
        x_fused = self.fusion_conv(x_cat)
        x_out = self.attention(x_fused)
        return x_out

class DAFNet(nn.Module):
    def __init__(self, dino_dim=384):
        super().__init__()

        self.adapter = MMSA_Encoder()
        self.bottleneck_conv = ConvBlock(dino_dim + 384, 512)

        self.dec3 = FusionDecoderBlock(512, 256, 256)
        self.dec2 = FusionDecoderBlock(256, 128, 128)
        self.dec1 = FusionDecoderBlock(128, 64, 64)

        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 112->224
            ConvBlock(64, 32)
        )

        self.head = nn.Conv2d(32, 4, kernel_size=3, padding=1)

    def forward(self, rgb, mask, alpha, fg_gray, dino_features):
        x_in = torch.cat([rgb, mask, alpha, fg_gray], dim=1)  #

        skips = self.adapter(x_in)
        c1, c2, c3, c4 = skips[0], skips[1], skips[2], skips[3]

        if dino_features.shape[-2:] != c4.shape[-2:]:
            dino_features = F.interpolate(dino_features, size=c4.shape[-2:], mode='bilinear')

        bottleneck = torch.cat([dino_features, c4], dim=1)  #
        x = self.bottleneck_conv(bottleneck)  # ->

        x = self.dec3(x, c3)  # -> 28
        x = self.dec2(x, c2)  # -> 56
        x = self.dec1(x, c1)  # -> 112
        x = self.final_up(x)  # -> 224

        out = self.head(x)  #

        pred_norm_raw = out[:, :3, :, :]
        pred_kappa_raw = out[:, 3:, :, :]

        pred_norm = F.normalize(pred_norm_raw, dim=1, p=2)
        pred_kappa = F.softplus(pred_kappa_raw) + 1e-6

        return {
            "pred_norm": pred_norm,  #
            "pred_kappa": pred_kappa  #
        }
