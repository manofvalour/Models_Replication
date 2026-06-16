import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.profiler import profile, record_function, ProfilerActivity


# data importing and processing
class DataProcessing:
    def __init__(self, text):
        self.txt = text
        chars = sorted(set(text))
        self.length = len(chars)

        self.stoi = {ch:i for i, ch in enumerate(chars)};
        self.itos = {i:ch for i, ch in enumerate(chars)};

    def encode_and_split(self, text, n):
        encode = lambda s:[self.stoi[c] for c in s]

        data = torch.tensor(encode(text), dtype=torch.long);
        if n:
            size = int(n * len(data))
            train_data, val_data = data[:size], data[size:]
            return train_data, val_data, self.length
        elif n ==0.0:
            return data, self.length

    def decode(self, idx):
        return ''.join(self.itos[i] for i in idx)


#loading data from memory
def load_text(filename):
    with open(filename, "r", encoding="utf-8") as f:
        text = f.read()

    return text

def get_batch(data, batch_size, seq_len):

    starts = torch.randint(
        0, len(data) - seq_len-1, size=(batch_size,)
    )

    X = torch.stack([data[i:i+seq_len] for i in starts])
    y = torch.stack([data[i+1:i+seq_len+1] for i in starts])

    return (X.clone().detach(),
           y.clone().detach())

  #Setting up the learning rate
def get_lr(it,config):
    if it< config.WARMUP_STEPS:
      return config.MAX_LR * (it+1)/config.WARMUP_STEPS
    if it > config.MAX_STEPS:
      return config.MIN_LR
    
    decay_ratio = (it - config.WARMUP_STEPS) / (config.MAX_STEPS - config.WARMUP_STEPS)
    assert 0<= decay_ratio <=1
    coef = 0.5 *(1.0 + math.cos(math.pi * decay_ratio))
    return config.MIN_LR + coef * (config.MAX_LR - config.MIN_LR)

@torch.no_grad()
def evaluate(model, data,batch_size, seq_len, device):
    model.eval()
    losses = []
    for _ in range(50):
        X, y = get_batch(data, batch_size, seq_len)
        X = X.to(device)
        y = y.to(device)

        with torch.amp.autocast('cuda', dtype = torch.float16):
            _, loss = model(X, y)
        losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)


##Getting the data batches
class DataLoaderLite:
    def __init__(self, config):
        import tiktoken
        self.B = config.BATCH_SIZE
        self.T = config.SEQ_LEN
        
        self.encoder = tiktoken.get_encoding('gpt2')
        self.current_position = 0

    
    def encode_and_split(self, text, n:int):
        tokens = self.encoder.encode(text)
        data = torch.tensor((tokens), dtype=torch.long)
        if n==0.0:
            print(f"Loaded {len(self.tokens)}tokens training data and {len(self.val_data)}tokens val data")
            print(f"Epoch = {len(self.tokens//(self.B*self.T))} batches")

            return data

        else:
            size = int(n * len(data))
            train_data, val_data = data[:size], data[size:]
    
            print(f"Loaded {len(train_data)}tokens training data and {len(val_data)}tokens val data")
            print(f"Epoch: {len(train_data//(self.B*self.T))} batches")

            return train_data, val_data
                  

    def next_batch(self, data):
        B,T = self.B, self.T

        buf = data[self.current_position: self.current_position+(B*T)+1]
        x = buf[:-1].view(B,T)
        y = buf[1:].view(B,T)

        self.current_position+= B*T

        if self.current_position + ((B*T)+1) > len(self.tokens):
            self.current_position=0

        return x,y
    
    def decode(self, idx):
        return self.encoder.decode(idx)