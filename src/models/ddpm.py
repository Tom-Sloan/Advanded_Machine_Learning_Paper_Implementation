import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels):
        super().__init__()
        self.time_mlp = nn.Linear(time_channels, out_channels)
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x, t):
        h = self.conv1(x)
        h = self.norm1(h)
        h += self.time_mlp(t)[:, :, None, None]
        h = F.silu(h)
        
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)
        
        return h + self.shortcut(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, model_channels=128, out_channels=3, 
                 num_res_blocks=2, attention_resolutions=(16,), 
                 dropout=0.1, channel_mult=(1, 2, 4, 8), conv_resample=True):
        super().__init__()
        
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            TimeEmbedding(model_channels),
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # Input convolution
        self.input_blocks = nn.ModuleList([
            nn.Conv2d(in_channels, model_channels, 3, padding=1)
        ])
        
        # Downsampling
        input_block_chans = [model_channels]
        ch = model_channels
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ConvBlock(ch, mult * model_channels, time_embed_dim)]
                ch = mult * model_channels
                input_block_chans.append(ch)
                self.input_blocks.append(nn.ModuleList(layers))
            if level != len(channel_mult) - 1:
                self.input_blocks.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
                input_block_chans.append(ch)
        
        # Middle block
        self.middle_block = ConvBlock(ch, ch, time_embed_dim)
        
        # Upsampling
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [ConvBlock(
                    ch + input_block_chans.pop(),
                    model_channels * mult,
                    time_embed_dim
                )]
                ch = model_channels * mult
                if level and i == num_res_blocks:
                    layers.append(nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1))
                self.output_blocks.append(nn.ModuleList(layers))
        
        # Output
        self.out = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1),
        )
    
    def forward(self, x, timesteps):
        # Time embedding
        emb = self.time_embed(timesteps)
        
        # Downsampling
        h = x
        hs = []
        for module in self.input_blocks:
            if isinstance(module, nn.ModuleList):
                for layer in module:
                    h = layer(h, emb)
            else:
                h = module(h)
            hs.append(h)
        
        # Middle
        h = self.middle_block(h, emb)
        
        # Upsampling
        for module in self.output_blocks:
            if isinstance(module, nn.ModuleList):
                for layer in module:
                    if isinstance(layer, ConvBlock):
                        h = torch.cat([h, hs.pop()], dim=1)
                        h = layer(h, emb)
                    else:
                        h = layer(h)
        
        return self.out(h)

def create_diffusion_model():
    """Creates and initializes the DDPM UNet model"""
    model = UNet(
        in_channels=3,
        model_channels=128,
        out_channels=3,
        num_res_blocks=2,
        attention_resolutions=(16,),
        dropout=0.1,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True
    )
    return model 