import os
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

DINOV3_GITHUB_LOCATION = r"dinov3"
if os.getenv("DINOV3_LOCATION") is not None:
    DINOV3_LOCATION = os.getenv("DINOV3_LOCATION")
else:
    DINOV3_LOCATION = DINOV3_GITHUB_LOCATION
print(f"DINOv3 location set to {DINOV3_LOCATION}")


MODEL_DINOV3_VITS = "dinov3_vits16" # 384
MODEL_DINOV3_VITS_CKPT = r"dinov3_vits16_pretrain_lvd1689m-08c60483.pth"

MODEL_NAME = MODEL_DINOV3_VITS
MODEL_CKPT_PATH = MODEL_DINOV3_VITS_CKPT

new_cache = r"checkpoints"
os.makedirs(new_cache, exist_ok=True)
torch.hub.set_dir(new_cache)

dino_model = torch.hub.load(
    repo_or_dir=DINOV3_LOCATION,
    model=MODEL_NAME,
    source="local",
    weight=MODEL_CKPT_PATH,
)
dino_model.cuda().eval()

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

MODEL_TO_NUM_LAYERS = {
    MODEL_DINOV3_VITS: 12,
}

n_layers = MODEL_TO_NUM_LAYERS[MODEL_NAME]

PATCH_SIZE = 16
IMAGE_SIZE = 512
# quantization filter for the given patch size
patch_quant_filter = torch.nn.Conv2d(1, 1, PATCH_SIZE, stride=PATCH_SIZE, bias=False)
patch_quant_filter.weight.data.fill_(1.0 / (PATCH_SIZE * PATCH_SIZE))


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        mid_channels = max(in_channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        attn = self.sigmoid(avg_out + max_out)  # (B,C,1,1)
        return attn

class TransFusionChannelAttn(nn.Module):
    def __init__(self,
                 unet_ch=512,
                 dinov3_ch=128,
                 mid_ch=128,
                 reduction=16):
        super().__init__()

        self.unet_down = nn.Conv2d(unet_ch, mid_ch, 1, bias=False)
        self.unet_up   = nn.Conv2d(mid_ch, unet_ch, 1, bias=False)

        self.q_conv = nn.Conv2d(mid_ch, mid_ch, 1, bias=False)
        self.k_conv = nn.Conv2d(dinov3_ch, mid_ch, 1, bias=False)
        self.v_conv = nn.Conv2d(dinov3_ch, mid_ch, 1, bias=False)

        self.ca_after_attn = ChannelAttention(mid_ch, reduction)

        cat_ch = unet_ch + dinov3_ch
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(cat_ch, cat_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(cat_ch),
            nn.ReLU(inplace=True)
        )
        self.ca_fuse = ChannelAttention(cat_ch, reduction)

        self.ls = nn.Parameter(torch.zeros(1, cat_ch, 1, 1))

    def forward(self, r4, aux):
        B, C_u, H, W = r4.shape
        _, C_a, _, _ = aux.shape

        u = self.unet_down(r4)

        q = self.q_conv(u).view(B, 128, -1).permute(0, 2, 1)
        k = self.k_conv(aux).view(B, 128, -1)
        v = self.v_conv(aux).view(B, 128, -1).permute(0, 2, 1)

        attn = torch.bmm(q, k) / (128 ** 0.5)
        attn = F.softmax(attn, dim=-1)
        u_refine = torch.bmm(attn, v)
        u_refine = u_refine.permute(0, 2, 1).view(B, 128, H, W)

        ca_weight = self.ca_after_attn(u_refine)
        u_refine = u_refine * ca_weight

        u = u + u_refine
        r4_refine = self.unet_up(u)

        fused = torch.cat([r4_refine, aux], dim=1)

        fused_out = self.fuse_conv(fused)
        fused_out = fused_out * self.ca_fuse(fused_out)
        fused = fused + self.ls * fused_out

        return fused

class TransformerBottleneck(nn.Module):
    def __init__(self, dim=640, n_layers=6, n_heads=8, dinov3_dim=128):
        super().__init__()
        self.dim = dim
        self.rgb_up = nn.Linear(512, dim)
        self.dino_proj = nn.Sequential(
            nn.Linear(dinov3_dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim)
        )
        self.ca_q = nn.Linear(dim, dim, bias=False)
        self.ca_kv = nn.Linear(dim, dim * 2, bias=False)
        self.ca_scale = dim ** -0.5
        self.ca_norm1 = nn.LayerNorm(dim)
        self.ca_norm2 = nn.LayerNorm(dim)
        self.ca_drop = nn.Dropout(0.1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=dim * 4,
            dropout=0.1, activation="relu", batch_first=False
        )
        self.tr = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.coord_mlp = nn.Sequential(
            nn.Linear(4, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim)
        )

        self.channel_proj = nn.Linear(640, 512)  # 640 -> 512

    def forward(self, feat: torch.Tensor):
        B, C, H, W = feat.shape
        N = H * W

        rgb_feat = feat[:, :512, :, :]
        dino_feat = feat[:, 512:, :, :]

        rgb_token = rgb_feat.view(B, 512, N).permute(2, 0, 1)
        dino_token = dino_feat.view(B, 128, N).permute(2, 0, 1)

        rgb_token = self.rgb_up(rgb_token)
        dino_token = self.dino_proj(dino_token)

        q = self.ca_q(self.ca_norm1(rgb_token))
        kv = self.ca_kv(self.ca_norm2(dino_token))
        k, v = kv.chunk(2, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.ca_scale
        attn = F.softmax(attn, dim=-1)
        ca_out = self.ca_drop(attn @ v)

        token = rgb_token + ca_out

        out = self.tr(token)

        tokens_out = out.permute(1, 2, 0).contiguous().view(B, C, H, W)

        tokens_out = tokens_out.permute(0, 2, 3, 1)
        tokens_out = self.channel_proj(tokens_out)
        tokens_out = tokens_out.permute(0, 3, 1, 2).contiguous()
        return tokens_out, None

class SimpleDecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class AuxEncoder(nn.Module):
    def __init__(self, in_ch=384, out_ch=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 2, 1, bias=False),  # 32->16
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class TransDINO(nn.Module):
    def __init__(self, backbone='resnet34', pretrained=True, embed_dim=16, device='cpu'):
        self.device = device
        super().__init__()
        # backbone
        if backbone == 'resnet34':
            res = models.resnet34(pretrained=pretrained)
            self.rgb_stem = nn.Sequential(res.conv1, res.bn1, res.relu, res.maxpool)
            self.layer1 = res.layer1
            self.layer2 = res.layer2
            self.layer3 = res.layer3
            self.layer4 = res.layer4
            rgb_chs = [64, 64, 128, 256, 512]
        else:
            raise ValueError('unsupported backbone')

        self.aux = AuxEncoder(in_ch=384, out_ch=128)

        self.fuse_deep = nn.Conv2d(rgb_chs[4] + 32 * 4, 512, 1)

        self.bottleneck = TransformerBottleneck(dim=640, n_layers=6, n_heads=8, dinov3_dim=128)

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec3 = SimpleDecoderBlock(512 + rgb_chs[3], 256)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = SimpleDecoderBlock(256 + rgb_chs[2], 128)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = SimpleDecoderBlock(128 + rgb_chs[1], 64)
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.mask_head = nn.Conv2d(32, 1, 1)
        self.embed_head = nn.Conv2d(32, embed_dim, 1)
        self.reffusion = TransFusionChannelAttn()

    def forward(self, rgb: torch.Tensor, aux: torch.Tensor):
        B, _, H, W = rgb.shape
        s = self.rgb_stem(rgb)
        r1 = self.layer1(s)
        r2 = self.layer2(r1)
        r3 = self.layer3(r2)
        r4 = self.layer4(r3)

        aux_feats = self.aux(aux)
        fused = self.reffusion(r4, aux_feats)
        tokens_out, prompts_out = self.bottleneck(fused)

        x = self.up3(tokens_out)
        x = torch.cat([x, r3], dim=1)
        x = self.dec3(x)

        x = self.up2(x)
        x = torch.cat([x, r2], dim=1)
        x = self.dec2(x)

        x = self.up1(x)
        x = torch.cat([x, r1], dim=1)
        x = self.dec1(x)

        x = self.final_conv(x)
        mask_logits = self.mask_head(x)

        return mask_logits

    def postprocess_mask(self, mask_logits: torch.Tensor, target_size: Tuple[int, int],
                         mode: str = 'bilinear', align_corners: bool = False) -> torch.Tensor:
        return F.interpolate(mask_logits, size=target_size, mode=mode, align_corners=align_corners)

    def preprocess_image(self, image: torch.Tensor, target_size: Tuple[int, int],
                         mode: str = 'bilinear', align_corners: bool = False) -> torch.Tensor:
        return F.interpolate(image, size=target_size, mode=mode, align_corners=align_corners)

