import torch
import torch.nn as nn
import torch.nn.functional as F

from model_architecture import Block

##creating out model with torch
class Transformer(nn.Module):
    def __init__(self, config, vocab_size, pad_token_id):
        super().__init__()
        self.config = config
        self.pad_token_id = pad_token_id

        self.tok_emb = nn.Embedding(vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        
        self.encoder = nn.ModuleList([Block(config, is_decoder=False) for _ in range(config.n_layer)])
        self.decoder = nn.ModuleList([Block(config, is_decoder=True) for _ in range(config.n_layer)])
        
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, vocab_size, bias=False)
        
        # Weight Tying
        self.tok_emb.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, src, tgt_input, targets=None):
        device = src.device
        
        # Encoder
        s_pos = torch.arange(src.size(1), device=device)
        x = self.tok_emb(src) + self.pos_emb(s_pos)
        for block in self.encoder: 
            x = block(x)

        enc_out = x

        # Decoder
        t_pos = torch.arange(tgt_input.size(1), device=device)
        y = self.tok_emb(tgt_input) + self.pos_emb(t_pos)
        for block in self.decoder: 
            y = block(y, enc_out=enc_out)
        
        logits = self.lm_head(self.ln_f(y))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.flatten(0, 1),#-1, logits.size(-1)), 
                targets.flatten(0), 
                ignore_index=self.pad_token_id,
                label_smoothing=0.1
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, src, max_len=50, sos_id=1, eos_id=2):
        self.eval()
        device = src.device
        # Encode src once
        s_pos = torch.arange(src.size(1), device=device)
        enc_out = self.tok_emb(src) + self.pos_emb(s_pos)
        for block in self.encoder: enc_out = block(enc_out)

        idx = torch.full((src.size(0), 1), sos_id, dtype=torch.long, device=device)
        for _ in range(max_len):
            idx_cond = idx[:, -self.config.block_size:]
            t_pos = torch.arange(idx_cond.size(1), device=device)
            y = self.tok_emb(idx_cond) + self.pos_emb(t_pos)
            for block in self.decoder: y = block(y, enc_out=enc_out)
            
            logits = self.lm_head(self.ln_f(y[:, -1, :]))
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
            idx = torch.cat((idx, next_id), dim=1)
            if (next_id == eos_id).all(): break
        return idx

