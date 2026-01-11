import torch
import torch.nn as nn

class GatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, activation=nn.ELU()):
        super(GatedConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * 2, kernel_size,
                              stride, padding, dilation, bias=True)
        self.activation = activation
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        feature, gate = torch.chunk(x, 2, dim=1)
        return self.activation(feature) * torch.sigmoid(gate)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x_reshaped = x.flatten(2).transpose(1, 2)
        residual = x_reshaped
        x_norm = self.norm1(x_reshaped)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x_reshaped = residual + attn_out
        residual = x_reshaped
        x_norm = self.norm2(x_reshaped)
        x_reshaped = residual + self.mlp(x_norm)
        x_out = x_reshaped.transpose(1, 2).reshape(B, C, H, W)
        return x_out

class TransDecomp(nn.Module):
    def __init__(self, in_channels=4, hidden_dim=64):
        super(TransDecomp, self).__init__()

        self.enc1 = GatedConv2d(in_channels, hidden_dim, kernel_size=5, padding=2, stride=1)
        self.enc2 = nn.Sequential(
            GatedConv2d(hidden_dim, hidden_dim * 2, stride=2),
            GatedConv2d(hidden_dim * 2, hidden_dim * 2)
        )
        self.enc3 = nn.Sequential(
            GatedConv2d(hidden_dim * 2, hidden_dim * 4, stride=2),
            GatedConv2d(hidden_dim * 4, hidden_dim * 4)
        )
        self.bottleneck_vit = nn.Sequential(
            TransformerBlock(dim=hidden_dim * 4, num_heads=4, mlp_ratio=4),
            TransformerBlock(dim=hidden_dim * 4, num_heads=4, mlp_ratio=4),
            TransformerBlock(dim=hidden_dim * 4, num_heads=4, mlp_ratio=4)
        )
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = GatedConv2d(hidden_dim * 4 + hidden_dim * 2, hidden_dim * 2)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = GatedConv2d(hidden_dim * 2 + hidden_dim, hidden_dim)
        self.head_alpha = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(hidden_dim // 2, 1, kernel_size=1)
        )
        self.head_foreground = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(hidden_dim // 2, 1, kernel_size=1)
        )

    def forward(self, rgb, mask):
        x_in = torch.cat([rgb, mask], dim=1)

        f1 = self.enc1(x_in)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)

        f_global = self.bottleneck_vit(f3)

        x = self.up3(f_global)
        x = torch.cat([x, f2], dim=1)
        x = self.dec3(x)

        x = self.up2(x)
        x = torch.cat([x, f1], dim=1)
        feat_out = self.dec2(x)

        raw_alpha = self.head_alpha(feat_out)
        pred_alpha = torch.sigmoid(raw_alpha)

        pred_alpha = pred_alpha * mask

        raw_fg = self.head_foreground(feat_out)
        pred_fg = torch.sigmoid(raw_fg)
        pred_fg = pred_fg * mask

        return pred_alpha, pred_fg
