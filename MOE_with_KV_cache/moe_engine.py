import torch
import torch.nn as nn
import torch.nn.functional as F

from moe_architecture import TransformerBlock


class MOEGPT(nn.Module):
    def __init__(self, config,  
                allow_moe:bool,
                moe_aux_loss_coef=0.01,
                ):
        super().__init__()
        self.config= config
        self.transformer = nn.ModuleDict(dict(
            token_embed_table = nn.Embedding(config.VOCAB_SIZE, config.DIM),
            pos_embed_table = nn.Embedding(config.SEQ_LEN, config.DIM),
            drop = nn.Dropout(config.DROPOUT),
            blocks = nn.ModuleList([TransformerBlock(config, config.DIM, config.N_HEAD, 
                                                    allow_moe) for _ in range(config.N_LAYER)]),
            ln_f = nn.LayerNorm(config.DIM),
        ))
        self.allow_moe = allow_moe
        self.lm_head = nn.Linear(config.DIM, config.VOCAB_SIZE, bias=False)
        
        self.apply(self._init_weights)
        for pn,p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / (2*config.N_LAYER)**0.5)


        #weight tying
        self.transformer.token_embed_table.weight = self.lm_head.weight
        self.moe_aux_loss= moe_aux_loss_coef

    # weight initialization
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # in GPT class
    def reset_cache(self):
        for block in self.transformer.blocks:
            block.attn.reset_cache()

    def forward(self, index, 
                target = None, 
                use_cache=False, 
                cache_pos=0):
        
        B,T = index.size()
        assert T<= self.config.SEQ_LEN, "Input sequence length exceed maximum sequence length"

        tok_emb = self.transformer.token_embed_table(index) 
        if use_cache and T == 1:
            # decode: use actual position in sequence
            pos = torch.tensor([cache_pos], device=index.device)
        else:
            # prefill or no cache: standard position range
            pos = torch.arange(T, device=index.device)

        pos_emb = self.transformer.pos_embed_table(pos)

        x = self.transformer.drop(tok_emb + pos_emb) 

        if self.allow_moe:
            aux_losses = []

            for blocks in self.transformer.blocks:
                x, aux_l = blocks(x, use_cache)
                aux_losses.append(aux_l)

            x = self.transformer.ln_f(x)
            logits = self.lm_head(x)

            if target is None:
                loss = None
            else:
                logits = logits.view(-1,logits.size(-1))
                targets = target.reshape(-1)
                loss = F.cross_entropy(logits, targets);

            aux_loss = torch.stack(aux_losses).mean() * self.moe_aux_loss

            return logits, loss, aux_loss
        
        else:
            for blocks in self.transformer.blocks:
                x = blocks(x, use_cache)

            x = self.transformer.ln_f(x)
            logits = self.lm_head(x)

            if target is None:
                loss = None
            else:
                logits = logits.view(-1,logits.size(-1))
                targets = target.reshape(-1)
                loss = F.cross_entropy(logits, targets);

            return logits, loss
    
        
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, 
                temperature = 1.0, 
                top_k = None, 
                use_cache=True
                ):
        
        if use_cache:
            self.reset_cache()
        
            self._cache_pos = idx.size(1)

            self(
                index=idx,
                use_cache=True,
                cache_pos=0
            )
        for _ in range(max_new_tokens):
            if use_cache:
            # ── DECODE: pass only the single new token ─────────────────
                if self._cache_pos >= self.config.SEQ_LEN:
                    break     

                idx_next_input = idx[:, -1:]          # (B, 1) — last token only
                if self.allow_moe:
                    logits, _,_ = self(
                        index=idx_next_input,
                        use_cache=True,
                        cache_pos=self._cache_pos
                    )

                    self._cache_pos += 1
                else:
                    logits, _ = self(
                        index=idx_next_input,
                        use_cache=True,
                        cache_pos=self._cache_pos
                    )

                    self._cache_pos += 1

            else:
                if self.allow_moe:

                    idx_cond = idx if idx.size(1) <= self.config.SEQ_LEN else idx[:, -self.config.SEQ_LEN:]
                    logits, _, _ = self(index=idx_cond, use_cache=False)
                
                else:
                    idx_cond = idx if idx.size(1) <= self.config.SEQ_LEN else idx[:, -self.config.SEQ_LEN:]
                    logits, _ = self(index=idx_cond, use_cache=False)

            logits = logits[:, -1, :] / temperature
           
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits<v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


