import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity
from model_architecture import TransformerBlock
from gpt2_config import GPT2Config
import inspect

class GPT2(nn.Module):
    def __init__(self, config,vocab_size):
        super().__init__()
        self.config=config
        self.transformer = nn.ModuleDict(dict(
            token_embed_table = nn.Embedding(vocab_size, config.DIM),
            pos_embed_table = nn.Embedding(config.SEQ_LEN, config.DIM),
            drop = nn.Dropout(config.DROPOUT),
            blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.N_LAYER)]),
            ln_f = nn.LayerNorm(config.DIM),
        ))

        self.lm_head = nn.Linear(config.DIM, vocab_size, bias=False)
        
        #weight tying
        self.transformer.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn,p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / (2*config.N_LAYER)**0.5)


    # weight initialization
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.01)


    def forward(self, index, target = None):
        B,T = index.size()
        assert T<= self.config.SEQ_LEN

        tok_emb = self.transformer.token_embed_table(index) 
        pos_emb = self.transformer.pos_embed_table(torch.arange(T, device =index.device))

        x = self.transformer.drop(tok_emb + pos_emb) 

        for blocks in self.transformer.blocks:
            x = blocks(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        if target is None:
            loss = None
        else:
            logits = logits.view(-1,logits.size(-1))
            targets = target.reshape(-1)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature = 1.0, top_k = None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.SEQ_LEN else idx[:, -self.config.SEQ_LEN:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]/temperature

            if top_k is not None:
                topk_probs, topk_indices = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
                #logits[logits<topk_probs[:, [-1]]] = float('-inf')
                ix = torch.multinomial(topk_probs, 1)
                xcol = torch.gather(topk_indices, -1, ix)
                idx = torch.cat((idx, xcol), dim=1)

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
    
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
    
    def configure_optimizers(self, weight_decay, learning_rate, device):
        ## start with all the candidate parameters (the requires_grad)
        param_dict = {pn:p for pn,p in self.named_parameters()}
        param_dict = {pn:p for pn,p in param_dict.items() if p.requires_grad}

        decay_params = [p for n,p in param_dict.items() if p.dim()>=2]
        nodecay_params = [p for n,p in param_dict.items() if p.dim() <2]

        optim_groups = [{'params': decay_params, 'weight_decay': weight_decay},
                        {'params': nodecay_params, "weight_decay": 0.0}]
        
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params} parameters")

        ## creating adamw optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and "cuda" in device
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9,0.95), eps=1e-8, fused=use_fused)

        return optimizer

