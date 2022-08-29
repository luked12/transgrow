"""
code adopted from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
"""
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
        self.dropout = nn.Dropout(p=dropout)
        
    def get_posEnc(self, time):
        return self.pe[time.long(), :]

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)