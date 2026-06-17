import torch
import torch.nn as nn
import torch.nn.functional as F

from lstm_architecture import LSTMLayer

    
class LSTM(nn.Module):
    def __init__(self,n_embd, vocab_size, 
                 n_hidden, n_layers, proj_size=0, 
                 dropout=0.0, bias=True):
        
        super().__init__()
        self.embd = nn.Embedding(vocab_size, n_embd)
        self.n_layers = n_layers
        self.dropout_prob = dropout
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        # Track the dimension of the data as it flows through layers
        current_layer_input = n_embd

        for i in range(n_layers):
            # Only the final layer gets the user-defined proj_size
            # (unless you want every layer projected)
            is_last_layer = (i == n_layers - 1)
            p_size = proj_size if is_last_layer else 0

            # Create the layer
            self.layers.append(
                LSTMLayer(current_layer_input, n_hidden, proj_size=p_size)
            )

            #The next layer's input size is this layer's output size
            #If we projected, the next layer sees proj_size. Otherwise, n_hidden.
            current_layer_input = p_size if p_size > 0 else n_hidden

        self.lm_head = nn.Linear(current_layer_input, vocab_size)

    def forward(self, x, y = None, hx=None):
        # hx: tuple of (h0, c0) each shaped [n_layers, batch, hidden/proj]
        if isinstance(x, (list, tuple)) and y is None:
            x,y = x

        x = self.embd(x)
        B,T,C = x.shape
        h_all, c_all = [], []
        current_input = x

        for i, layer in enumerate(self.layers):
            # Extract specific state for this layer if provided
            layer_hx = None
            if hx is not None:
                layer_hx = (hx[0][i], hx[1][i])

            # Pass through the LSTMLayer (the batch/time loop happens inside here)
            out_seq, (hn, cn) = layer(current_input, layer_hx)

            h_all.append(hn)
            c_all.append(cn)

            # Apply Dropout between layers (but not after the last layer)
            if self.dropout_prob > 0 and i < self.n_layers - 1:
                current_input = self.dropout(out_seq)
                #F.dropout(out_seq, p=self.dropout_prob, training=self.training)
            else:
                current_input = out_seq

        #logit = current_input.view(B,T, -1)
        logit = self.lm_head(current_input) # (B, T, vocab_size)
        #print(f"final_out: {logit.shape}") # 4x8x65

        if y!= None:
            loss = F.cross_entropy(logit.view(-1, logit.size(-1)), y.view(-1))

            return logit, (h_all, c_all), loss

        else:
            return logit, (h_all, c_all)


    def generate(self, idx, n_output=100, topk=10):

        for _ in range(n_output):
            idx_cond = idx[:, -1:]
            logits, _ = self(idx_cond)
            logits = logits[:,-1,:]

            probs = F.softmax(logits, dim=1)
            topk_probs, _ = torch.topk(probs, topk, dim =-1)
            topk_idx = torch.multinomial(topk_probs, num_samples =1)

           # idx_next = torch.gather(topk_indices, -1, ix)

            idx = torch.cat((idx, topk_idx), dim=1)

        return idx

     # topk_probs, topk_indices = torch.topk(probs, 50, dim =-1)
    #   ix = torch.multinomial(probs, 1, generator = sample_rng) #(8,1)
        # gather the corresponding indices
       # xcol = torch.gather(topk_indices, -1, ix) #B,1
        #append to the sequence


