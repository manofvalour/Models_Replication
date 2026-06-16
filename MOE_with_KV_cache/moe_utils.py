import torch
import torch.nn as nn
import torch.nn.functional as F



# data importing and processing
class DataProcessing:
    def __init__(self, text):
        self.txt = text
        chars = sorted(set(text))
        self.length = len(chars)

        self.stoi = {ch:i for i, ch in enumerate(chars)}
        self.itos = {i:ch for i, ch in enumerate(chars)}

    def encode_and_split(self, text, n):
        encode = lambda s:[self.stoi[c] for c in s]

        data = torch.tensor(encode(text), dtype=torch.long)
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

@torch.no_grad()
def evaluate(model, config, val_data, device):

    model.eval()
    losses = []

    for _ in range(50):
        X, y = get_batch(val_data, config.BATCH_SIZE, config.SEQ_LEN)

        X = X.to(device)
        y = y.to(device)

        with torch.amp.autocast('cuda', dtype = torch.float16):
            _, loss, _ = model(X, y)
        losses.append(loss.item())

    model.train()

    return sum(losses) / len(losses)
