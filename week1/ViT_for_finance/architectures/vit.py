import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from timm.models.layers import trunc_normal_

import torch
from torch import nn
import numpy as np

class EmbeddingLayer(nn.Module):
    def __init__(self, patch_size: int, in_channels: int, d_model: int, grid_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.grid_size = grid_size
        self.num_patches = (grid_size // patch_size) ** 2
        self.proj = nn.Linear(patch_size * patch_size * in_channels, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, d_model))  

        trunc_normal_(self.pos_embed, std=.02)

    def forward(self, x):
        batch_size, height, width, channels = x.shape

        patches = x.unfold(1, self.patch_size, self.patch_size).unfold(2, self.patch_size, self.patch_size)
        
        patches = patches.permute(0, 2, 3, 1, 4, 5)  # Reorder to (batch_size, num_patches_height, num_patches_width, channels, patch_size, patch_size)
        patches = patches.contiguous().view(batch_size, -1, self.patch_size * self.patch_size * channels)  # Flatten patches

        # Project the patches into the embedding space
        patch_embeddings = self.proj(patches)

        # Add the class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, d_model)
        embeddings = torch.cat((cls_tokens, patch_embeddings), dim=1)  # (batch_size, num_patches + 1, d_model)

        # Add positional embedding
        embeddings = embeddings + self.pos_embed

        return embeddings

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        assert d_model % num_heads == 0
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        attn_probs = nn.functional.softmax(attn_scores, dim=-1)
        
        attn_output = torch.matmul(attn_probs, V)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        out = self.out_proj(attn_output)
        
        return out


class MLP(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))

class Block(nn.Module):
    def __init__(self, d_model: int, num_heads: int, mlp_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 =nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, mlp_dim, dropout_rate)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)


    def forward(self, x):
        attn_out = self.attention(self.norm1(x))
        x = x + self.dropout1(attn_out)
        mlp_out = self.mlp(self.norm2(x))
        x = x + self.dropout2(mlp_out) 
        return x

class VisionTransformer(nn.Module):
    def __init__(self, patch_size: int = 13, in_channels: int = 1, d_model: int = 64, num_heads: int = 8, 
                 mlp_dim: int = 128, num_layers: int = 6, num_classes: int = 10, dropout_rate: float = 0.1):
        super().__init__()
        self.embedding = EmbeddingLayer(patch_size, in_channels, d_model, grid_size=65)
        self.blocks = nn.ModuleList([
            Block(d_model, num_heads, mlp_dim, dropout_rate) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.mlp_head = nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        cls_token_output = x[:, 0]  # Class token output
        out = self.mlp_head(cls_token_output)
        return out