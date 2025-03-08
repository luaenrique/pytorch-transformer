import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        # https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
        # the idea here is to generate vectors of size d_model to represent words 
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # this formula comes from the article Attention is All You Need:  
        # "In the embedding layers, we multiply those weights by √dmodel."
        return self.embedding(x) * math.sqrt(self.d_model)
        
class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model: int, seq_length: int, dropout: float) -> None:
        super.__init__()
        self.d_model = d_model
        self.seq_length = seq_length
        self.dropout = dropout

        # starting by defining a positional encoding matrix considering the model dimension and 
        # sequence length
        positional_encoding = torch.zeros(self.d_model, self.seq_length)

        # create a vector to represent the word in the sentence
        # unsqueeze() The . unsqueeze() method in PyTorch adds a new dimension of size one at the specified position in a tensor
        position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1) # (seq_len, 1)


        # now, we have to apply a formula described in the article (Attention Is All You Need)
        # for odd positions: PE(pos, 2i) = sin(pos/10000^(2i/dmodel))
        # for even positions: PE(pos, 2 + i) = cos(pos/10000^(2i/dmodel))
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model))

        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)

        positional_encoding = positional_encoding.unsqueeze(0)

        self.register_buffer('positional_encoding', positional_encoding)

    def forward(self, x):
        x = x + (self.positional_encoding[:, :x.shape[1], :]).requires_grad(False)
        return self.dropout(x)
    


class LayerNormalization(nn.Module):
    # Epsilon is used here to avoid the x become so big when sigma is almost zero
    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # Multiplied
        self.bias = nn.Parameter(torch.zeros(1)) # Added

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True)
        std = x.std(dim = -1, keepdim=True)

        return self.alpha (x - mean) / (std + self.eps) + self.bias