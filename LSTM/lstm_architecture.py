import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMCell(nn.Module):
    def __init__(self, n_input, n_hidden, proj_size=0, bias=True):
        super().__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.proj_size = proj_size
        self.use_bias = bias

        # Combine all 4 gates (i, f, g, o) into one matrix for efficiency
        #initializing kaiming he
        self.W_ig = nn.Parameter(torch.randn(n_input, 4 * n_hidden) *n_input**-0.5) #1/sqrt(input_size)
        #If projecting, the hidden state coming back in is of proj_size
        h_in_dim = proj_size if proj_size > 0 else n_hidden

        self.W_hig = nn.Parameter(torch.randn(h_in_dim, 4 * n_hidden) * h_in_dim**-0.5)

        if bias:
            self.b = nn.Parameter(torch.zeros(4 * n_hidden))
        else:
            self.register_parameter('b', None)

        if proj_size > 0:
            self.W_prj = nn.Parameter(torch.randn(n_hidden, proj_size) * n_hidden**-0.5)
        else:
            self.register_parameter('W_prj', None)

    def forward(self, x, h_init, c_init):
        """
            x: [batch_size, seq_len]
            h_init: [batch_size, proj_size/n_hidden]
            c_init: [batch_size, n_hidden]
        """

        gates = (x @ self.W_ig) + (h_init @ self.W_hig)
        if self.use_bias:
            gates += self.b

        # Split into the 4 gates
        i, f, g, o = gates.chunk(4, dim=1)

        ig = torch.sigmoid(i)
        fg = torch.sigmoid(f)
        gg = torch.tanh(g) # often called 'g' or 'c_tilde' candidate memory
        og = torch.sigmoid(o)

        # Cell state update
        ct = (fg * c_init) + (ig * gg)
        # Hidden state update
        ht = og * torch.tanh(ct)

        if self.proj_size > 0:
            ht = ht @ self.W_prj

        return ht, ct
    
class LSTMLayer(nn.Module):
    def __init__(self, n_input, n_hidden, proj_size=0):
        super().__init__()
        self.cell = LSTMCell(n_input, n_hidden, proj_size)
        self.n_hidden = n_hidden
        self.proj_size = proj_size
        self.h_dim = proj_size if proj_size > 0 else n_hidden

    def forward(self, x, states=None):
        """
        Args:
            x (torch.tensor): input (Batch_size, seq_len, dim)
            states (optional): Defaults to None.

        Returns: 
        """

        batch_size, seq_len, _ = x.shape

        if states is None:
            h_t = torch.zeros(batch_size, self.h_dim, device=x.device)
            c_t = torch.zeros(batch_size, self.n_hidden, device=x.device)
        else:
            h_t, c_t = states

        outputs = []
        # Loop through TIME, not through batches
        for t in range(seq_len):
            x_t = x[:, t, :] # Get current time step for ALL batches
            h_t, c_t = self.cell(x_t, h_t, c_t)

            outputs.append(h_t.unsqueeze(1))
        output_seq = torch.cat(outputs, dim=1)
        return output_seq, (h_t, c_t)
