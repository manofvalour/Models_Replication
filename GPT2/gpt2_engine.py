import torch
import torch.nn.functional as F
import torch.nn as nn
import math

from gpt2_config import GPT2Config

## Multihead Attention Block
class CausalAttnBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd%config.n_head == 0

        self.c_attn = nn.Linear(config.n_embd, config.n_embd*3)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_embd = config.n_embd
        self.n_head = config.n_head

        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size)).view(
                                        1,1, config.block_size, config.block_size))

    def forward(self, x):
        B,T,C = x.size()
        qkv = self.c_attn(x) 
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.view(B,T, self.n_head, C//self.n_head).transpose(1,2)
        k = k.view(B,T, self.n_head, C//self.n_head).transpose(1,2)
        v = v.view(B,T, self.n_head, C//self.n_head).transpose(1,2)

        attn = (q@k.transpose(-2,-1))*(1/math.sqrt(k.size(-1)))
        attn = attn.masked_fill(self.bias[:,:,:T,:T]==0, float('-inf'))
        attn = F.softmax(attn, dim = -1)
        y = attn@v
        #y = F.scaled_dot_product_attention(q,k,v, is_causal=True)
        y = y.transpose(1,2).contiguous().view(B,T,C)

        y = self.c_proj(y)

        return y


## Feed Forward Network
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')

    def forward(self, x):
        y = self.c_proj(self.gelu(self.c_fc(x)))

        return y

## GPT Block (Attn subblock and FF sub block)
class GPTBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attn = CausalAttnBlock(config)
        self.mlp = MLP(config)
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.ln_2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        return x

class GPT2(nn.Module):
    def __init__(self, config:GPT2Config):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([GPTBlock(config) for _ in range(config.n_layers)]),
            ln_f = nn.LayerNorm(config.n_embd))
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias= False)

        ## weight tieing
        self.lm_head.weight = self.transformer.wte.weight
        self.apply(self._weight_init)

    def _weight_init(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: 
                nn.init.zeros_(module.weight)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.01)

    def forward(self, x, y=None):
        assert len(x.shape)>=2, "shape should be B,T,C"
        B,T = x.shape
        assert T<= self.config.block_size
        pos = torch.arange(0,T, dtype=torch.long, device=x.device)

        embd_x = self.transformer.wte(x) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            embd_x = block(embd_x)

        logit = self.lm_head(self.transformer.ln_f(embd_x))
        loss = None

        if y != None:
            loss = F.cross_entropy(
                logit.flatten(0,1),
                y.flatten(0),
                label_smoothing=0.1
            ) 

        return logit, loss
    
    @classmethod
    def pretrained(cls, model_type):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print(f"loading weight from pretrained gpt: {model_type}")

        config_args = {
            'gpt2': dict(n_layers=12, n_head = 12, n_embd=768),
            'gpt2-medium': dict(n_layers=24, n_head= 16, n_embd=1024),
            'gpt2-large': dict(n_layers=36, n_hed=20, n_embd=1280),
            'gpt2-xl': dict(n_layers=48, n_head=25, n_embd=1600)
        }[model_type]

        config_args['vocab_size']=50257 #vocabulary size
        config_args['block_size']=1024 #seq_len

        ################# LOCAL CUSTOM MODEL ###################
        config = GPT2Config(**config_args)
        model = GPT2(config)
        sd = model.state_dict()
        sd_key = sd.keys()
        sd_key = [k for k in sd_key if not k.endswith('.attn.bias')]

        ########################### HUGGINGFACE GPT2 MODEL ###################################
        ## init model from hugging face 
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_key_hf = sd_hf.keys()
        sd_key_hf = [k for k in sd_key_hf if not k.endswith('.attn.masked_bias')]
        sd_key_hf = [k for k in sd_key_hf if not k.endswith('.attn.bias')]
        
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        assert len(sd_key_hf)==len(sd_key), f"mismatch keys: {len(sd_key_hf)!= len(sd_key)}"
        for k in sd_key_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape

                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())

            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model