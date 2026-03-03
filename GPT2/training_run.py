from gpt2_config import GPT2Config
from gpt2_engine import GPT2

import os
import torch
import torch.nn as nn
import torch.nn.function as F
import time

num_return_seq = 5
max_entries=30

device = 'cpu'
if torch.cuda.is_available():
  device = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
  device = 'mps'
print(f"using device: {device}")

## getting the data batch
import tiktoken

B,T = 16,1024
class DataLoaderLite:
  def __init__(self, B,T):
    import tiktoken
    self.B = B
    self.T = T

    with open('input.txt', 'r') as f:
        text = f.read()
    
    encoder = tiktoken.get_encoding('gpt2')
    tokens = encoder.encode(text)
    self.tokens = torch.tensor(tokens)

    print(f"Loaded {len(self.tokens)} tokens")
    print(f"Epoch = {len(self.tokens//(B*T))} batches")

    self.current_position = 0

  def next_batch(self):
    B,T = self.B, self.T

    buf = self.tokens[self.current_position: self.current_position+(B*T)+1]
    x = buf[:-1].view(B,T)
    y = buf[1:].view(B,T)

    self.current_position+= B*T

    if self.current_position + ((B*T)+1) > len(self.tokens):
      self.current_position=0

    return x,y


## model training
torch.manual_seed(1337)
if torch.cuda.is_available():
  torch.cuda.manual_seed(1337)

train_loader = DataLoaderLite(B,T)
model = GPT2(GPT2Config())
model.to(device)
model = torch.compile(model)

## training loop
optimizer = torch.optim.AdamW(model.parameters(), lr = 3e-4)
for i in range(50):
  t0 = time.time()
  x,y = train_loader.next_batch()
  x,y = x.to(device), y.to(device)

  optimizer.zero_grad()
  with torch.autocast(device_type=device, dtype=torch.bfloat16):
    logits, loss = model(x,y)

    #import code; code.interact(local=locals())

  loss.backward()
  optimizer.step()
  torch.cuda.synchronize()

  t1 = time.time()
  dt = (t1-t0)*1000 #ms
  toks_per_sec = [train_loader.B * train_loader.T]/(t1-t0)

  print(f"step(i) | loss: {loss.item():.2f} | dt: {dt:.2f}ms | toks/sec: {toks_per_sec}")



import sys; sys.exit()

############# MODEL GENERATION ################
#model = GPT2.pretrained('gpt2')
torch.set_float32_matmul_precision('high') #for gpu to use tf32 for matmul ops
model = GPT2(GPT2Config())
model.eval()
model.to(device)

# prefill phase
import tiktoken
enc= tiktoken.get_encoding('gpt2')
tokens= enc.encode("Hello, I'm a Language model,")
tokens = torch.tensor(tokens, dtype = torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_seq, 1)
x = tokens.to(device)
print(x.shape)

## the prediction phase
torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1)< max_entries:
  with torch.no_grad():
    logits, _ = model(x)
    logits = logits[:, -1, :]
    probs = F.softmax(logits, dim=-1)
    topk_probs, topk_indices = torch.topk(probs, k=50,dim=-1)
    ix = torch.multinomial(topk_probs, 1)
    xcol = torch.gather(topk_indices, -1, ix)
    x = torch.cat((x, xcol), dim=1)

for i in range(num_return_seq):
  tokens = x[i, :max_entries].tolist()
  dec = enc.decode(tokens)
  print(">", dec)



