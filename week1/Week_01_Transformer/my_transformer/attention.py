import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from typing import Optional, Tuple

class QueryLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super(QueryLayer, self).__init__()
        self.linear = nn.Linear(d_model, d_model * n_heads)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class KeyLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super(KeyLayer, self).__init__()
        self.linear = nn.Linear(d_model, d_model * n_heads)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class ValueLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super(ValueLayer, self).__init__()
        self.linear = nn.Linear(d_model, d_model * n_heads)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class ScaledDotProductAttention(nn.Module):
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # what are the dimensions of this matrix?
        # I need to think about this for a bit

        # Need to think about the dimensions of the query, key, and value matrix
        # One thing we know for 
        numerator = torch.matmul(q, k.transpose(-2,-1))
        fraction = numerator / torch.sqrt(torch.tensor(k.size()[-1], dtype = torch.float32))
        if mask is not None: 
            fraction = fraction.masked_fill(mask == 0, float('-inf'))
        weights = F.softmax(fraction, dim=-1)

        output = torch.matmul(weights, v)
        return output, weights



class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        
        self.query_layers = QueryLayer(d_model, n_heads)
        self.key_layers = KeyLayer(d_model, n_heads)
        self.value_layers = ValueLayer(d_model, n_heads)
        self.attention = ScaledDotProductAttention()
        self.fc = nn.Linear(n_heads * d_model, d_model)
    
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = Q.size()[0]

        query_1 = self.query_layers(Q)
        key_1 = self.key_layers(K)
        value_1 = self.value_layers(V)

        query = query_1.view(batch_size, -1, self.n_heads, (self.d_model//self.n_heads)).transpose(1,2)
        key = key_1.view(batch_size, -1, self.n_heads, (self.d_model//self.n_heads)).transpose(1,2)
        value = value_1.view(batch_size, -1, self.n_heads, (self.d_model//self.n_heads)).transpose(1,2)

        updated_value, weights = self.attention(query, key, value, mask)

        updated_value = updated_value.transpose(1,2).contiguous().view(batch_size, -1, self.n_heads * self.d_model)
        
        output = self.fc(updated_value)

        return output