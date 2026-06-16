import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalMultiheadAttn(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.N_HEAD
        self.head_size = config.HEAD_SIZE
        DIM_TOTAL = config.DIM

        self.qkv = nn.Linear(config.DIM, 3 * DIM_TOTAL, bias=False)
        self.proj = nn.Linear(DIM_TOTAL, config.DIM)
        self.attn_dropout = nn.Dropout(config.DROPOUT)
        self.res_dropout = nn.Dropout(config.DROPOUT)

        self.flash = hasattr(F, 'scaled_dot_product_attention') ##return bool
        if not self.flash:
            self.register_buffer(
                'mask', 
                torch.tril(torch.ones(config.SEQ_LEN, config.SEQ_LEN))
                .view(1,1, config.SEQ_LEN, config.SEQ_LEN))

        self.cache_k = None
        self.cache_v = None

    def reset_cache(self):
        self.cache_k = None
        self.cache_v = None

    def forward(self, x, use_cache=False):
        B,T,C = x.shape
        qkv = self.qkv(x)

        q,k,v = qkv.chunk(3, dim=-1)
        q = q.view(B,T,self.n_heads, self.head_size).transpose(1,2)
        k = k.view(B,T,self.n_heads, self.head_size).transpose(1,2)
        v = v.view(B,T,self.n_heads, self.head_size).transpose(1,2)
        
        if use_cache:
            if self.cache_k is None:

                self.cache_k = k
                self.cache_v = v

                out = F.scaled_dot_product_attention(
                    q,k,v,
                    is_causal=True,
                    dropout_p=self.attn_dropout.p if self.training else 0
                )

            else:
                self.cache_k = torch.cat(
                    [self.cache_k,k], dim=2
                )

                self.cache_v = torch.cat(
                    [self.cache_v,v], dim=2
                )

                out = F.scaled_dot_product_attention(
                    q,
                    self.cache_k,
                    self.cache_v,
                    is_causal=False,
                    dropout_p=0
                )
              
        else:
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

class Router(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.TOP_K
        self.layer = nn.Linear(config.DIM, config.NUM_EXPERT)

    def forward(self, x):
        weights = F.softmax(self.layer(x), dim=-1)
        topk_vals, topk_idx = torch.topk(weights, self.top_k, dim=1)
        topk_vals_norm = topk_vals/topk_vals.sum(dim=1, keepdim=True)

        return weights, topk_vals_norm, topk_idx


class Expert(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.dropout = nn.Dropout(config.DROPOUT)
        self.input_proj = nn.Linear(config.DIM, config.DIM*4)
        self.output_proj = nn.Linear(config.DIM*4, config.DIM)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.input_proj(x)
        x = self.act(x)
        x = self.output_proj(x)

        return self.dropout(x)

class MoELayer(nn.Module):
    def __init__(self, config):
                
        super().__init__()
        assert 1<=config.TOP_K <=config.NUM_EXPERT, "top_k must be between 1 and num_experts"

        self.top_k = config.TOP_K
        self.dropout = nn.Dropout(config.DROPOUT)
        self.num_experts = config.NUM_EXPERT
        self.router= Router(config)
        self.experts = nn.ModuleList([Expert(config) for _ in range(config.NUM_EXPERT)])
    
    def load_balance_loss(self, router_probs, top_k_indices):
        total_tokens, total_experts = router_probs.shape
        importance = router_probs.mean(dim =0)
        all_selected_experts = top_k_indices.reshape(-1)
        load = (
            torch.bincount(all_selected_experts, minlength=total_experts)
            .float()/(total_tokens * self.top_k)
        )

        loss = total_experts * (importance * load).sum()

        return loss

    def forward(self,x):
        B,T,C = x.shape
        x_flattened = x.reshape(B*T, C)

        router_probs, top_k_weights, top_k_indices = self.router(x_flattened)
        output = torch.zeros_like(x_flattened)

        for expert_id in range(self.num_experts):
            mask = (top_k_indices == expert_id)

            if not mask.any():
                continue

            token_ids, k_positions = mask.nonzero(as_tuple=True)

            expert_input = x_flattened[token_ids]
            expert_output = self.experts[expert_id](expert_input)
            weights = top_k_weights[token_ids, k_positions].unsqueeze(-1)

            output[token_ids] += expert_output * weights
        aux_loss = self.load_balance_loss(router_probs, top_k_indices)
        output = output.reshape(B, T, C)

        return self.dropout(output), aux_loss
    
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
    def __init__(self, config, n_embed, 
                 n_head, allow_moe:bool):
        super().__init__()
        head_size = n_embed//n_head
        self.attn= CausalMultiheadAttn(config)

        if allow_moe:
            self.moe = MoELayer(config)
        else:
            self.mlp = MLP(config)

        self.allow_moe = allow_moe

        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x, use_cache):
        x = x + self.attn(self.ln1(x), use_cache = use_cache)

        if self.allow_moe:
            moe_output, aux_loss = self.moe(self.ln2(x))
            x = x+moe_output
            return x, aux_loss
        
        else:
            mlp_output = self.mlp(self.ln2(x))
            x = x + mlp_output
            return x

