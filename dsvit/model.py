import torch
import torch.nn as nn

# --------------------------
# Pre-Encoder Module
# --------------------------
class PreEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)

# --------------------------
# Multi-Scale Token Embedder
# --------------------------
class MultiScaleEmbedder(nn.Module):
    def __init__(self, in_channels=16, embed_dim=64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=4, stride=4),
            nn.ReLU()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)                      # (B, D, H/4, W/4)
        return x.flatten(2).transpose(1, 2)   # (B, N, D)

# --------------------------
# Dense Transformer Block
# --------------------------
class DenseTransformerBlock(nn.Module):
    def __init__(self, dim, heads=4, mlp_dim=128):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, prev_feats=[]):
        for feat in prev_feats:
            x = x + feat  # dense skip connections
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.mlp(x))
        return x

# --------------------------
# DSViT Detection Model
# --------------------------
class DSViTDetector(nn.Module):
    def __init__(self, num_classes=3, embed_dim=64, depth=4):
        super().__init__()
        self.pre_encoder = PreEncoder()
        self.embedder = MultiScaleEmbedder(in_channels=16, embed_dim=embed_dim)
        
        self.transformer_blocks = nn.ModuleList([
            DenseTransformerBlock(embed_dim) for _ in range(depth)
        ])

        self.shared_backbone_dim = embed_dim
        self.token_pool = nn.AdaptiveAvgPool1d(1)  # Pool token dim → global

        # Detection Heads
        self.bbox_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 4)  # x, y, w, h
        )
        self.class_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)  # classification logits
        )

    def forward(self, x):
        feat = self.pre_encoder(x)        # (B, 16, H, W)
        tokens = self.embedder(feat)      # (B, N, D)

        prev_feats = []
        for block in self.transformer_blocks:
            tokens = block(tokens, prev_feats)
            prev_feats.append(tokens)

        pooled = tokens.mean(dim=1)       # global token pooling (B, D)

        bbox_pred = self.bbox_head(pooled)     # (B, 4)
        class_logits = self.class_head(pooled) # (B, num_classes)

        return bbox_pred, class_logits


# --------------------------
# ViT Baseline - No PreEncoder, No MultiScale, No Dense
# --------------------------
class ViTBaseline(nn.Module):
    def __init__(self, num_classes=3, embed_dim=64, depth=4):
        super().__init__()
        self.patch_embed = nn.Conv2d(1, embed_dim, kernel_size=4, stride=4)
        self.transformer_blocks = nn.ModuleList([
            DenseTransformerBlock(embed_dim) for _ in range(depth)
        ])
        self.token_pool = nn.AdaptiveAvgPool1d(1)

        self.bbox_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 4)
        )
        self.class_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        x = self.patch_embed(x)
        tokens = x.flatten(2).transpose(1, 2)
        for block in self.transformer_blocks:
            tokens = block(tokens)
        pooled = tokens.mean(dim=1)
        return self.bbox_head(pooled), self.class_head(pooled)

# --------------------------
# DSViT - No PreEncoder
# --------------------------
class DSViT_NoPreEncoder(nn.Module):
    def __init__(self, num_classes=3, embed_dim=64, depth=4):
        super().__init__()
        self.embedder = MultiScaleEmbedder(in_channels=1, embed_dim=embed_dim)
        self.transformer_blocks = nn.ModuleList([
            DenseTransformerBlock(embed_dim) for _ in range(depth)
        ])
        self.token_pool = nn.AdaptiveAvgPool1d(1)

        self.bbox_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 4)
        )
        self.class_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        tokens = self.embedder(x)
        prev_feats = []
        for block in self.transformer_blocks:
            tokens = block(tokens, prev_feats)
            prev_feats.append(tokens)
        pooled = tokens.mean(dim=1)
        return self.bbox_head(pooled), self.class_head(pooled)

# --------------------------
# DSViT - No MultiScale Embedder (uses raw patches)
# --------------------------
class DSViT_NoMultiScale(nn.Module):
    def __init__(self, num_classes=3, embed_dim=64, depth=4):
        super().__init__()
        self.pre_encoder = PreEncoder()
        self.patch_embed = nn.Conv2d(16, embed_dim, kernel_size=4, stride=4)
        self.transformer_blocks = nn.ModuleList([
            DenseTransformerBlock(embed_dim) for _ in range(depth)
        ])
        self.token_pool = nn.AdaptiveAvgPool1d(1)

        self.bbox_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 4)
        )
        self.class_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        feat = self.pre_encoder(x)
        tokens = self.patch_embed(feat).flatten(2).transpose(1, 2)
        prev_feats = []
        for block in self.transformer_blocks:
            tokens = block(tokens, prev_feats)
            prev_feats.append(tokens)
        pooled = tokens.mean(dim=1)
        return self.bbox_head(pooled), self.class_head(pooled)
