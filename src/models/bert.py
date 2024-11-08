import torch
import torch.nn as nn
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]

class BERT(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, 
                 dim_feedforward=2048, dropout=0.1, max_len=5000):
        super().__init__()
        
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.position_encoding = PositionalEncoding(d_model, max_len)
        
        # Layer normalization and dropout
        self.embed_dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)
        
        # MLM head
        self.mlm_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size)
        )
        
        self.d_model = d_model
        self._init_weights()
        
    def _init_weights(self):
        # Initialize embeddings and linear layers
        nn.init.normal_(self.token_embedding.weight, mean=0, std=0.02)
        for module in self.mlm_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
    def forward(self, src, attention_mask=None):
        # Create attention mask if provided
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None
            
        # Embed tokens and add positional encoding
        x = self.token_embedding(src) * math.sqrt(self.d_model)
        x = self.position_encoding(x)
        x = self.embed_dropout(x)
        x = self.layer_norm(x)
        
        # Pass through transformer with attention mask
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        # Get MLM predictions
        mlm_output = self.mlm_head(x)
        
        return mlm_output

def create_bert_base(vocab_size):
    """Creates an improved BERT base model"""
    model = BERT(
        vocab_size=vocab_size,
        d_model=512,        # Increased from 256
        nhead=8,
        num_layers=6,
        dim_feedforward=2048,  # Increased from 1024
        dropout=0.1
    )
    return model