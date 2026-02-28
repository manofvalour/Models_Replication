import torch
import torch.nn.functional as F
import torch.nn as nn
from dataclasses import dataclass
import math

## Multihead Attention Block
class CausalAttnBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd%config.n_head == 0

        self.attn_wghts = nn.Linear(config.n_embd, config.n_embd*3)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_embd = config.n_embd
        self.n_head = config.n_head

        self.register_buffer('tril', torch.tril(torch.ones(config.seq_len, config.seq_len)).view(
                                        1,1, config.seq_len, config.seq_len))

    def forward(self, x):
        B,T,C = x.size()
        qkv = self.attn_wghts(x) 
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.view(B,T, self.n_head, C//self.n_head).transpose(1,2)
        k = k.view(B,T, self.n_head, C//self.n_head).transpose(1,2)
        v = v.view(B,T, self.n_head, C//self.n_head).transpose(1,2)

        attn = (q@k.transpose(-2,-1))*(1/math.sqrt(k.size(-1)))
        attn = attn.masked_fill(self.bias[:,:,:T,:T]==0, float('-inf'))
        attn = F.softmax(attn, dim = -1)
        y = attn@v
        y = y.transpose(1,2).contiguous().view(B,T,C)
        y = self.out_proj(y)

        return y


## Feed Forward Network
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.lm1 = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.lm2 = nn.Linear(4 * config.n_embd, config.vocab_size)
        self.gelu = nn.GELU(approximate='tanh')

    def forward(self, x):
        y = self.lm2(self.gelu(self.lm1(x)))

        return y

## GPT Block (Attn subblock and FF sub block)
class GPTBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mha = CausalAttnBlock(config)
        self.ml_percept = MLP(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.ml_percept(self.ln2(x))

        return x

class GPT2(nn.module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(
            tok_embd = nn.Embedding(config.vocab_size, config.n_embd),
            pos_embd = nn.Embedding(config.seq_len, config.n_embd),
            gpt_block = nn.ModuleList([GPTBlock(config) for _ in range(config.n_layers)]),
            ln = nn.LayerNorm(config.n_embd)
        )
        self.lm = nn.Linear(config.n_embd, config.vocab_size, bias= False)

        ## weight tieing
        self.tok_embd.weight = self.lm.weight
        self.apply(self._weight_init)

    def _weight_init(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: 
                nn.init.zeros_(module.weight)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.01)

    def forward(self, x, y=None):
        assert len(x.shape)>=3, "shape should be B,T,C"
        B,T,C = x.shape

        embd_x = self.tok_embd(x) + self.pos_embd(x)
        y_pred = self.gpt_block(embd_x, y)

        logit = self.lm(self.ln(y_pred))
        loss = None

        if y != None:
            logit, loss = self.lm(self.ln(y_pred))

        return logit, loss