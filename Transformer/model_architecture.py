import torch
import torch.nn as nn
import torch.nn.functional as F


#===================================THE TRANSFORMER MODEL ARCHITECTURE ====================================
class MultiHeadAttention(nn.Module):
    def __init__(self, config, is_causal=False):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_size = config.n_embd // config.n_head
        self.is_causal = is_causal

        # Efficient single-matrix projections
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.dropout = config.dropout

    def forward(self, x, kv_input=None):
        B, T, C = x.shape
        kv_input = kv_input if kv_input is not None else x
        Tk = kv_input.size(1)

        # Reshape to (B, nh, T, hs)
        q = self.q_proj(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = self.k_proj(kv_input).view(B, Tk, self.n_head, self.head_size).transpose(1, 2)
        v = self.v_proj(kv_input).view(B, Tk, self.n_head, self.head_size).transpose(1, 2)

        # Handles scaling
        y = F.scaled_dot_product_attention(
            q, k, v, 
            dropout_p=self.dropout if self.training else 0.0, 
            is_causal=self.is_causal and kv_input is x
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
            x = x + self.attn(self.ln1(x))
            x = x + self.cross_attn(self.ln2(x), kv_input=enc_out)
            x = x + self.ffwd(self.ln3(x))
        else:
            x = x + self.attn(self.ln1(x))
            x = x + self.ffwd(self.ln2(x))
        return x