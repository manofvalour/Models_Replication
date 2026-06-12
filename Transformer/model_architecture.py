import torch
import torch.nn as nn
import torch.nn.functional as F
import math


#===================================THE TRANSFORMER MODEL ARCHITECTURE ====================================
class MultiHeadAttention(nn.Module):
    def __init__(self, config, is_causal=False, is_cross=False):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.config=config
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_size = config.n_embd // config.n_head
        self.is_causal = is_causal
        self.is_cross = is_cross
        self.dropout = nn.Dropout(config.dropout)

        # Efficient single-matrix projections
        if is_cross:
            self.q_proj  = nn.Linear(config.n_embd, config.n_embd, bias=False)
            self.kv_proj = nn.Linear(config.n_embd, 2*config.n_embd, bias=False)
        else:
            self.attn_w = nn.Linear(config.n_embd, 3*config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, x, kv_input=None):
        B, T, C = x.shape
        kv_input = kv_input if kv_input is not None else x
        Tk = kv_input.size(1)

        if self.is_cross:
            q = self.q_proj(x)
            k, v = self.kv_proj(kv_input).split(self.n_embd, dim=-1) #cross attn block
        else:
            q, k, v = self.attn_w(x).split(self.n_embd, dim=-1) #decoderblock

        q = q.reshape(B, q.size(1),  self.n_head, self.head_size).transpose(1, 2)
        k = k.reshape(B, k.size(1), self.n_head, self.head_size).transpose(1, 2)
        v = v.reshape(B, v.size(1), self.n_head, self.head_size).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=self.is_causal
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


## transformer block
class Block(nn.Module):
    def __init__(self, config, is_decoder=False):
        super().__init__()
        self.is_decoder = is_decoder
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        
        self.attn = MultiHeadAttention(config, is_causal=is_decoder)
        self.ffwd = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

        if is_decoder:
            self.cross_attn = MultiHeadAttention(config, is_causal=False)
            self.ln3 = nn.LayerNorm(config.n_embd)

    def forward(self, x, enc_out=None):
        # Pre-LayerNorm structure for better gradient flow
        if self.is_decoder:
            x = self.ln1(x + self.attn(x))
            x = self.ln2(x + self.cross_attn(x, kv_input=enc_out))
            x = self.ln3(x + self.ffwd(x))
        else:
            x = self.ln1(x + self.attn(x))
            x = self.ln2(x + self.ffwd(x))
        return x