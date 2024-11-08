import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=True, use_dropout=False):
        super().__init__()
        self.downsample = downsample
        
        if downsample:
            self.conv = nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)
        else:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False)
            
        self.norm = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(0.5) if use_dropout else None
        
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        if self.downsample:
            x = F.leaky_relu(x, 0.2)
        else:
            x = F.relu(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x

class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        
        # Encoder
        self.e1 = nn.Conv2d(in_channels, 64, 4, 2, 1)  # 128
        self.e2 = UNetBlock(64, 128)    # 64
        self.e3 = UNetBlock(128, 256)   # 32
        self.e4 = UNetBlock(256, 512)   # 16
        self.e5 = UNetBlock(512, 512)   # 8
        self.e6 = UNetBlock(512, 512)   # 4
        self.e7 = UNetBlock(512, 512)   # 2
        self.e8 = nn.Conv2d(512, 512, 4, 2, 1)  # 1
        
        # Decoder with skip connections
        self.d1 = UNetBlock(512, 512, False, True)    # 2
        self.d2 = UNetBlock(1024, 512, False, True)   # 4
        self.d3 = UNetBlock(1024, 512, False, True)   # 8
        self.d4 = UNetBlock(1024, 512, False)         # 16
        self.d5 = UNetBlock(1024, 256, False)         # 32
        self.d6 = UNetBlock(512, 128, False)          # 64
        self.d7 = UNetBlock(256, 64, False)           # 128
        self.d8 = nn.ConvTranspose2d(128, out_channels, 4, 2, 1)  # 256
        
    def forward(self, x):
        # Encoder
        e1 = F.leaky_relu(self.e1(x), 0.2)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)
        e8 = F.leaky_relu(self.e8(e7), 0.2)
        
        # Decoder with skip connections
        d1 = self.d1(e8)
        d2 = self.d2(torch.cat([d1, e7], 1))
        d3 = self.d3(torch.cat([d2, e6], 1))
        d4 = self.d4(torch.cat([d3, e5], 1))
        d5 = self.d5(torch.cat([d4, e4], 1))
        d6 = self.d6(torch.cat([d5, e3], 1))
        d7 = self.d7(torch.cat([d6, e2], 1))
        d8 = self.d8(torch.cat([d7, e1], 1))
        
        return torch.tanh(d8)

class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=6):  # 6 channels for concatenated input
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(256, 512, 4, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(512, 1, 4, 1, 1)  # Output 30x30 PatchGAN
        )
        
    def forward(self, x):
        # Modified to accept concatenated input directly
        return self.layers(x)

def init_weights(model):
    """Initialize network weights using normal distribution"""
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and 'Conv' in classname:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
            
    model.apply(init_func)
    return model

def create_pix2pix_models():
    """Creates and initializes the Pix2Pix generator and discriminator"""
    generator = init_weights(Generator())
    discriminator = init_weights(PatchDiscriminator())
    return generator, discriminator 