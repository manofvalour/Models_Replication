import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

def load_data(data_dir:str):
    with open(data_dir, 'r', encoding='utf-8') as f:
        text = f.read()

    return text

class TextDataset:
    def __init__(self, data):

        chars = sorted(list(set(data)))

        self.stoi = {ch:i for i,ch in enumerate(chars)}
        self.itos = {i:ch for i,ch in enumerate(chars)}

        self.vocab_size = len(chars)


    def encode_and_split_data(self, data, train_size:float=0.0):

        encode = lambda x: [self.stoi[c] for c in x]
        enc_data = torch.tensor(encode(data), dtype=torch.long)    

        ## spliting the data into train and val
        if train_size ==0.0:
            return enc_data, self.vocab_size
        
        else:
            n1 = int(train_size * len(enc_data))
            train_data = enc_data[:n1]
            val_data = enc_data[n1:]


            print(f"==> Data successfully split into train and validation data with train_size: {len(train_data)}, val_size: {len(val_data)}")
            
            return train_data, val_data, self.vocab_size


    def decode(self, idx):
        return "".join(self.itos[i] for i in idx)
    
def get_batch(data, batch_size, seq_len):

    starts = torch.randint(
        0, len(data) - seq_len-1, size=(batch_size,)
    )

    X = torch.stack([data[i:i+seq_len] for i in starts])
    y = torch.stack([data[i+1:i+seq_len+1] for i in starts])

    return (X.clone().detach(),
           y.clone().detach())

#Create simple Sub-Datasets for the DataLoader
class ChunkDataset(Dataset):
    def __init__(self, data_idx, 
                 seq_len):
        
        self.data = data_idx
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self,idx):
        
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len]

        return (x,y)


@torch.no_grad()
def evaluate(model, data,batch_size, seq_len, device):
    model.eval()
    losses = []
    for _ in range(50):
        X, y = get_batch(data, batch_size, seq_len)
        X = X.to(device)
        y = y.to(device)

        with torch.amp.autocast('cuda', dtype = torch.float16):
         _,_, loss = model(X, y)
        losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)


def estimate_loss(model, loader, device, eval_iters=10):
    """ Helper to get a stable loss estimate without running the whole dataset """
    model.eval()
    losses = torch.zeros(eval_iters)

    # Use a temporary iterator to grab a few batches
    data_iter = iter(loader)

    with torch.no_grad():
        for k in range(eval_iters):
            try:
                X, Y = next(data_iter)
            except StopIteration: # If we hit the end, restart
                data_iter = iter(loader)
                X, Y = next(data_iter)

            X, Y = X.to(device), Y.to(device)
            _, _, loss = model(X, y=Y)
            losses[k] = loss.item()

    model.train()
    return losses.mean()



    """
    
        checkpoint = {
        'step': idx + 1, # Use current epoch index + 1 for epoch count
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizers.state_dict(),
        'scheduler': scheduler_cos.state_dict(),

        'loss': loss.item(), # This `loss` is from the last training batch
        'val_loss': final_epoch_val_loss_scalar, # Store the scalar value
        # Optional: include the RNG state to ensure reproducibility
        'rng_state': torch.get_rng_state(),
        }

        # Save to a file if current validation loss is the best so far
        if final_epoch_val_loss_scalar < best_model_val_loss:
            best_model_val_loss = final_epoch_val_loss_scalar
            torch.save(checkpoint, f"/root/best_checkpoint_epoch_{idx}.pt")
            tqdm.write(f"Epoch {idx} | Best model saved with Val Loss: {best_model_val_loss:.4f}")

        # Append losses to lists for plotting
        lossi.append(loss.log10().item()) # log10 of last training batch loss
        val_loss.append(current_val_loss_epoch_end.log10().item()) # log10 of epoch's final val loss
        
    """