import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class PatchEmbedding(nn.Module):
    """
    Splits images into patches and linearly embeds them
    """
    def __init__(self, image_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        num_patches = (image_size // patch_size) ** 2
        
        # Linear projection of flattened patches
        self.projection = nn.Sequential(
            # Break image into patches and embed
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                     p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, embed_dim)
        )
        
        # Learnable classification token and position embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))

    def forward(self, x):
        b = x.shape[0]
        x = self.projection(x)
        # Add classification token
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat([cls_tokens, x], dim=1)
        # Add position embeddings
        x = x + self.pos_embedding
        return x

class MultiHeadAttention(nn.Module):
    """
    Multi-head self attention mechanism
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Linear layers for queries, keys, values
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        b, n, c = x.shape
        # Split into query, key, value
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), qkv)
        
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        # Apply attention to values
        x = attn @ v
        x = rearrange(x, 'b h n d -> b n (h d)')
        x = self.proj(x)
        return x

class MLP(nn.Module):
    """
    Multilayer perceptron used in transformer blocks
    """
    def __init__(self, in_features, hidden_features, out_features, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, out_features),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    """
    Transformer block with self-attention and MLP
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio=4., dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), embed_dim, dropout)

    def forward(self, x):
        # Self attention block with residual connection
        x = x + self.attn(self.norm1(x))
        # MLP block with residual connection
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) model
    """
    def __init__(self, image_size=224, patch_size=16, in_channels=3, 
                 num_classes=1000, embed_dim=768, depth=12, 
                 num_heads=12, mlp_ratio=4., dropout=0.):
        super().__init__()
        
        # Patch embedding layer
        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        
        # Stack of transformer blocks
        self.transformer = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) 
              for _ in range(depth)]
        )
        
        # Final layer normalization and classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)
        
        # Transformer blocks
        x = self.transformer(x)
        
        # Classification head
        x = self.norm(x)
        x = x[:, 0]  # Use [CLS] token only
        x = self.head(x)
        return x

def create_vit_base():
    """Creates a ViT-Base model as specified in the paper"""
    return VisionTransformer(
        image_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.1
    ) 