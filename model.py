import torch
import torch.nn as nn
import torch.nn.functional as F


class IlluminationEstimator(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(IlluminationEstimator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, 3, padding=1),
            nn.Sigmoid(),  # Illumination map in range [0, 1]
        )

    def forward(self, x):
        feat = self.encoder(x)
        illum = self.decoder(feat)
        return illum


class IGMSA(nn.Module):
    """
    Illumination-Guided Multi-Head Self-Attention
    """
    def __init__(self, dim, num_heads):
        super(IGMSA, self).__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, illum_map):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        illum_guidance = illum_map.reshape(B, 1, -1)
        attn = attn + illum_guidance

        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super(TransformerBlock, self).__init__()
        self.attention = IGMSA(dim, num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x, illum_map):
        x = x + self.attention(self.norm1(x), illum_map)
        x = x + self.ffn(self.norm2(x))
        return x


class IlluminationGuidedTransformer(nn.Module):
    def __init__(self, in_channels=3, num_heads=4, num_layers=4, dim=64):
        super(IlluminationGuidedTransformer, self).__init__()
        self.input_proj = nn.Conv2d(in_channels, dim, 1)
        self.transformer_layers = nn.ModuleList(
            [TransformerBlock(dim, num_heads) for _ in range(num_layers)]
        )
        self.output_proj = nn.Conv2d(dim, in_channels, 1)

    def forward(self, x, illum_map):
        B, C, H, W = x.shape
        x = self.input_proj(x)
        x = x.flatten(2).transpose(1, 2)

        illum_flat = illum_map.flatten(2)

        for layer in self.transformer_layers:
            x = layer(x, illum_flat)

        x = x.transpose(1, 2).view(B, -1, H, W)
        x = self.output_proj(x)
        return x


class Retinexformer(nn.Module):
    def __init__(self, in_channels=3, num_heads=4, num_layers=4, dim=64):
        super(Retinexformer, self).__init__()
        self.illum_estimator = IlluminationEstimator(in_channels, 1)
        self.igt = IlluminationGuidedTransformer(in_channels, num_heads,
                                                 num_layers, dim)

    def forward(self, x):
        illum_map = self.illum_estimator(x)
        enhanced = x / (illum_map + 1e-5)
        restored = self.igt(enhanced, illum_map)
        return restored.clamp(0, 1), illum_map


if __name__ == "__main__":
    model = Retinexformer()
    dummy_input = torch.rand(1, 3, 256, 256)
    output, illum = model(dummy_input)
    print("Output shape:", output.shape)
    print("Illumination map shape:", illum.shape)
