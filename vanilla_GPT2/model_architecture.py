import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalMultiheadAttn(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.N_HEAD
        self.head_size = config.HEAD_SIZE
        DIM_TOTAL = config.HEAD_SIZE * config.N_HEAD

       # self.attn_heads = nn.ModuleList([AttentionHead(head_size) for _ in range(num_heads)]);
        self.qkv = nn.Linear(config.DIM, 3 * DIM_TOTAL, bias=False)
        self.proj = nn.Linear(DIM_TOTAL, config.DIM)
        self.attn_dropout = nn.Dropout(config.DROPOUT)
        self.res_dropout = nn.Dropout(config.DROPOUT)

        self.flash = hasattr(F, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer(
                'mask', 
                torch.tril(torch.ones(config.SEQ_LEN, config.SEQ_LEN))
                .view(1,1, config.SEQ_LEN, config.SEQ_LEN))


    def forward(self, x):
        B,T,C = x.shape
        qkv = self.qkv(x)

        q,k,v = qkv.chunk(3, dim=-1)
        q = q.view(B,T,self.n_heads, self.head_size).transpose(1,2)
        k = k.view(B,T,self.n_heads, self.head_size).transpose(1,2)
        v = v.view(B,T,self.n_heads, self.head_size).transpose(1,2)
        
        if self.flash:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, 
                dropout_p=self.attn_dropout.p if self.training else 0)
        else:
            weights = (q @ k.transpose(-2,-1)) * (k.size(-1)**-0.5)
            weights = weights.masked_fill(self.mask[:, :, :T, :T]==0, float("-inf"))
            weights = self.attn_dropout(weights.softmax(dim=-1))

            out = weights @ v
        
        out = out.transpose(1,2).contiguous().view(B,T,C)
        out = self.res_dropout(self.proj(out))
     
        return out



class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(config.DIM, 4*config.DIM),
                    nn.GELU(), nn.Linear(4*config.DIM, config.DIM),)
        self.dropout = nn.Dropout(config.DROPOUT)

    def forward(self, x):
        x = self.net(x)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn= CausalMultiheadAttn(config)

        self.ff = MLP(config)
        self.ln1 = nn.LayerNorm(config.DIM)
        self.ln2 = nn.LayerNorm(config.DIM)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ff(self.ln2(x))

