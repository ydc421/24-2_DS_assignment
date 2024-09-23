import torch
import torch.nn as nn
import math
from torch import Tensor

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        """
        How does nn.Embedding work?
        Simplistic Explanation 
        We have vocab_size of 500 where it's represented by a 1-D tensor where each element is described by a token ID 
        Each token ID is used to create a vector of size d_model to represent that specific token ID
        """
    
    def forward(self, x: Tensor) -> Tensor:
        return self.embedding(x)

"""
Same nn.Embedding is used for PositionEmbedding to learn about this!
Remember it's not as simple as 1-D, 2D, we usually receive multiple batches of data = Within in batch, there is an input phrase, each phrase has multitude of words, 
where each word is represented by the embedding vector
"""


class PositionEmbedding(nn.Module): #One thing to note is that we're probably using relative positional embedding for better scalability
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super(PositionEmbedding, self).__init__() 
        """
        max_len represents the max length of our sequence of text, if it's too large, extremely costly to compute global dependencies!
        There are two methods of performing Position Embedding: 
            Absolute Positional Embedding with this code: 
                self.embedding = nn.Embedding(max_len, d_model)
            Sinusoidal Positional Embedding that the paper implemented! 
        """
        # First one here is just absolute embedding, but from my understanding, it can be used freely. 
        # This is going to be my implementation of the sinusoidal positional embedding matrix
        pos_emb = torch.zeros(max_len, d_model) # X axis = the number of tokens, Y axis = length of each individual embedding vector

        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1) # Turns this into a (d_model, 1)
        expon = (torch.arange(start= 0, end = d_model, step=2).float() / d_model)
        ratio = torch.pow(10000, expon)

        pos_emb[:,0::2] = torch.sin(position/ratio) # (d_model, d_model/2) where each vector is represented by a different dimension
        pos_emb[:,1::2] = torch.cos(position/ratio)
        # https://discuss.pytorch.org/t/what-is-the-difference-between-register-buffer-and-register-parameter-of-nn-module/32723
        # Code is based on this implementation: https://discuss.pytorch.org/t/what-does-register-buffer-do/121091
        self.register_buffer('pos_emb', pos_emb.unsqueeze(0))
        # register_buffer = for constants, kept part of a model's state, but non-trainable, which makes sense for this code since we're not training this embedding any further


    
    def forward(self, x: Tensor) -> Tensor:
        #TODO one line! 
        # Question is? Are we adding the positional embeddings already? 
        # I think it would be better for us to add the positional encoding to the original tensor since we're already using self.register_buffer!
        return self.pos_emb[:, :x.size(1), :]