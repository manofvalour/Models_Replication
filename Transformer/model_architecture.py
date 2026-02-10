import torch
import torch.nn as nn
import torch.nn.functional as F


#===================================THE TRANSFORMER MODEL ARCHITECTURE ====================================


# implementing Self-Attention head for the encoder part of the transformer
class SelfAttentionHead(nn.Module):
    def __init__(self, config,):
        super().__init__()
        self.config=config
        self.key = nn.Linear(config.n_embd, config.head_size, bias=False)             ##head_size is the dimension of the key (dk)
        self.query = nn.Linear(config.n_embd, config.head_size, bias =False)          ##head_size is the dimension of the query (dq)
        self.value = nn.Linear(config.n_embd, config.head_size, bias = False)         ##head_size is the dimension of the value (dv)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self,x, context =None):
        """
        x: main input data
        context: y (target dataset)

        """
        B,T,C = x.shape #batch, timestamp, dimension
        dk = self.config.head_size

        # computing the attention score
        kv_input = context if context != None else x
        q = self.query(x) #B,T,head_size
        k = self.key(kv_input)
        v = self.value(kv_input)

        wei = q @ k.transpose(-2,-1) * dk**-0.5       #(B,T,head_size) @ (B,head_size,T) --> (B,T,T)
        wei_norm = F.softmax(wei,dim=-1)
        wei_norm = self.dropout(wei_norm)

        ## perform the weighted aggregation of the value
        out = wei_norm @ v
        
        #out = F.scaled_dot_product_attention(q,k,v, dropout_p=self.config.dropout, is_causal = False) #flash attention
      

        return out
    

# implementing Self-Attention head with masking for the decoder part of the transformer
class MaskedSelfAttentionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.key = nn.Linear(config.n_embd, config.head_size, bias=False)             ##head_size is the dimension of the key (dk)
        self.query = nn.Linear(config.n_embd, config.head_size, bias =False)          ##head_size is the dimension of the query (dq)
        self.value = nn.Linear(config.n_embd, config.head_size, bias = False)         ##head_size is the dimension of the value (dv)

        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B,T,C = x.shape
        dk = self.config.head_size

        k = self.key(x)       #key from the encoder output
        q = self.query(x)     #query from the decoder input
        v = self.value(x)     #B,T,16    #value from the encoder output

        wei = q @ k.transpose(-2,-1) * dk**-0.5                      #(B,T,head_size) @ (B,head_size,T) --> (B,T,T)

        wei_masked = wei.masked_fill(self.tril[:T,:T]==0, float('-inf'))
        wei_norm = F.softmax(wei_masked, dim=-1)                           #normalized

        #out = F.scaled_dot_product_attention(q,k,v, dropout_p=self.config.dropout, is_causal = True) #flash attention
        wei_norm = self.dropout(wei_norm)

        ## perform the weighted aggregation of the value
        out = wei_norm @ v

        return out
    


## multihead attention
class MultiHeadAttention(nn.Module):
    def __init__(self, config, masked = False):
        super().__init__()
        self.masked = masked

        if masked is True:
            self.dec_heads = nn.ModuleList([MaskedSelfAttentionHead(config) for _ in range(config.n_head)])
        else:
            self.enc_heads = nn.ModuleList([SelfAttentionHead(config) for _ in range(config.n_head)])

        self.proj = nn.Linear(config.n_embd, config.n_embd)   #linear
        self.dropout= nn.Dropout(config.dropout)

    def forward(self, x, enc_out=None):
        if enc_out is not None:
            # cross attn head
            assert self.masked != True, "This should be a cross_attention head and should not be masked"
            out = torch.cat([h(x, enc_out) for h in self.enc_heads], dim = -1) #concatinating the output of the corss_attn heads
            out = self.dropout(self.proj(out))

        #decoder attn head
        elif enc_out is None and self.masked ==True:
            out = torch.cat([h(x) for h in self.dec_heads], dim = -1) #concatinating the output of masked attention heads in the decoder
            out = self.dropout(self.proj(out))

        else:
            # encoder attn head
            assert enc_out is None and self.masked!=True
            out = torch.cat([h(x, enc_out) for h in self.enc_heads], dim = -1) #concatinating the output of enc attn heads
            out = self.dropout(self.proj(out))

        return out
    

## feed forward network
class FeedForward(nn.Module):
    """a simple linear layer followed by non linearity"""

    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4*config.n_embd),
            nn.ReLU(),
            nn.Linear(4*config.n_embd, config.n_embd),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        return self.net(x)
    

## transformer block
class Block(nn.Module):
    def __init__(self, config, is_decoder = False):
        super().__init__()
        self.decoder =is_decoder
        if is_decoder:
            self.msk_attn = MultiHeadAttention(config, masked=True)     ## masked_attention head
            self.cross_attn = MultiHeadAttention(config)                ## cross_attention head
            self.ffwd = FeedForward(config)
            self.ln1 = nn.LayerNorm(config.n_embd)
            self.ln2 = nn.LayerNorm(config.n_embd)
            self.ln3 = nn.LayerNorm(config.n_embd)

        else:
            self.s_attn = MultiHeadAttention(config)                    ## self_attention head
            self.ffwd = FeedForward(config)
            self.ln1 = nn.LayerNorm(config.n_embd)
            self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x, enc_out=None):
        if enc_out !=None:
            #decoder block
            assert self.decoder is True, f"This is not a decoder block, accepts only one parameter"
            x= self.ln1(x + self.msk_attn(x))
            x = self.ln2(x + self.cross_attn(x, enc_out))
            x= self.ln3(x+ self.ffwd(x))

        else:
            #encoder block
            x= self.ln1(x + self.s_attn(x))
            x = self.ln2(x+self.ffwd(x))

        return x