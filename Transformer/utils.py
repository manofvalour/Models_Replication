import torch
import torch.nn as nn
import torch.nn.functional as F
import sacrebleu
import random
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

def clean_wmt_row(row, tokenizer, block_size=32)-> dict[dict[str, list[int] | int | bool]]:

    """Helper function to clean and tokenize a single row of the WMT dataset."""
    try:
        src_text = row['prompt'][0]['content'].replace('Translate this into French:', '').strip()
        tgt_text = row['completion'][0]['content'].strip()

        # Encoding the source and target text into token ids
        src_ids = tokenizer.encode(src_text, add_special_tokens=True)
        tgt_ids = tokenizer.encode(tgt_text, add_special_tokens=True)

        return {
            'src_ids': src_ids,
            'tgt_ids': tgt_ids,
            'src_len': len(src_ids),
            'tgt_len': len(tgt_ids),
            'keep': len(src_ids) <= block_size and len(tgt_ids) <= block_size
        }

    except Exception as e:
        # Mark as keep=False so we can filter out corrupted rows
        return {'src_ids': [], 'tgt_ids': [], 'src_len': 0, 'tgt_len': 0, 'keep': False}


import random

class DistributedWMTBatchSampler(torch.utils.data.Sampler):
    def __init__(self, sampler, dataset, max_tokens_per_batch=2048):
        self.sampler = sampler 
        self.dataset = dataset
        self.max_tokens = max_tokens_per_batch
        self.item_lens = np.maximum(
            np.array(dataset['src_len']), 
            np.array(dataset['tgt_len'])
        )

    def __iter__(self):
        indices = []
        for idx in self.sampler:
            if isinstance(idx, dict):
                raise TypeError(f"Sampler yielded a dict instead of an int. Value: {idx}")
            indices.append(int(idx))
        
        indices.sort(key=lambda i: self.item_lens[i])

        batches = []
        current_batch = []
        max_len = 0

        for idx in indices:
            item_len = self.item_lens[idx]
            temp_max_len = max(max_len, item_len)

            if (len(current_batch) + 1) * temp_max_len <= self.max_tokens:
                current_batch.append(idx)
                max_len = temp_max_len
            else:
                batches.append(current_batch)
                current_batch = [idx]
                max_len = item_len

        if current_batch:
            batches.append(current_batch)

        random.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self):
        return len(self.sampler) // (self.max_tokens // 40)

def collate_fn(batch):
    # batch is now a list of dictionaries from the Dataset
    src = [torch.tensor(item['src_ids'], dtype=torch.long) for item in batch]
    tgt = [torch.tensor(item['tgt_ids'], dtype=torch.long) for item in batch]

    src_padded = torch.nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=0)
    tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt, batch_first=True, padding_value=0)

    return src_padded, tgt_padded


## Evaluating BLEU Score
def calculate_bleu(model, dataloader, tokenizer, device, max_new_tokens=128):
    """
    Helper function to calculate the BLEU score on a subset of the validation set."""

    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        # We only evaluate a subset (e.g., 10 batches) to keep training fast
        for i, (src, tgt) in enumerate(dataloader):
            if i > max_new_tokens: break

            src = src.to(device)

            # model.generate now returns decoded strings, not token IDs
            decoded_preds = model.generate(src, max_new_tokens=max_new_tokens)
            decoded_tgts = tokenizer.batch_decode(tgt, skip_special_tokens=True)

            all_preds.extend(decoded_preds)
            all_targets.extend(decoded_tgts)

    bleu = sacrebleu.corpus_bleu(all_preds, [[ref] for ref in all_targets])
    return bleu.score


def estimate_loss(model, dataloader, eval_iters=20, device='cuda'):
    """
    Computes a stable average loss over a fixed number of batches.
    Works across multiple GPUs in DDP.
    """
    model.eval()
    
    # We use a local tensor to track sums so we can synchronize later
    total_loss = torch.tensor(0.0, device=device)
    count = torch.tensor(0.0, device=device)
    
    # Create a fresh iterator
    #data_iter = iter(dataloader)

    with torch.no_grad():
        for _ in range(eval_iters):
            try:
                # Grab next batch
                src, tgt = next(iter(dataloader))
            except StopIteration:
                # Restart if validation set is smaller than eval_iters
              #  data_iter = iter(dataloader)
                src, tgt = next(dataloader)

            src, tgt = src.to(device), tgt.to(device)
            # Alignment for translation (Shifted right for teacher forcing)
            tgt_input, tgt_label = tgt[:, :-1], tgt[:, 1:]

            # Use autocast to match your training precision (prevents dtype errors)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                _, loss = model(src, tgt_input, tgt_label)
            
            total_loss += loss
            count += 1

    # Calculate average on this GPU
    avg_loss = total_loss / count

    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(avg_loss, op=torch.distributed.ReduceOp.SUM)
        avg_loss = avg_loss / torch.distributed.get_world_size()

    model.train()
    return avg_loss


def get_transformer_schedule(optimizer, d_model, warmup_steps=4000):
    """
    Returns a learning rate scheduler based on the 'Attention is All You Need' formula.
    """

    def lr_lambda(step):
        # step + 1 to avoid division by zero at step 0
        step = step + 1

        term1 = step ** -0.5
        term2 = step * (warmup_steps ** -1.5)

        # Scale by d_model^-0.5
        return (d_model ** -0.5) * min(term1, term2)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(model, optimizer,
                    scheduler, train_loss,
                    val_loss,epoch, path):
    """Saves the model and optimizer state to a checkpoint file."""


    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss.item(),
        'rng_state': torch.get_rng_state(),
    }

    torch.save(checkpoint, path)
    print(f"Checkpoint saved at epoch {epoch} to {path}")


def load_checkpoint(model, optimizer, scheduler, path, device):
    """Loads the model and optimizer state from a checkpoint file."""

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    torch.set_rng_state(checkpoint['rng_state'])

    print(f"Checkpoint loaded from {path} at epoch {checkpoint['epoch']}")

    return (model, optimizer, scheduler,
            checkpoint['epoch'], checkpoint['train_loss'],
            checkpoint['val_loss'])


def data_prep(dataset, tokenizer, block_size=128, max_tokens_per_batch=2048, num_workers=4):
    ## cleaning, and spliting the dataset into train and test split
    split_ds = dataset['train'].train_test_split(test_size=0.2, seed=42)
    train_d = split_ds['train']
    val_d = split_ds['test']

    train_data = train_d.map(clean_wmt_row, fn_kwargs={"block_size":block_size, "tokenizer": tokenizer})
    filtered_tds = train_data.filter(lambda x: x['keep'])

    val_data = val_d.map(clean_wmt_row, fn_kwargs={"block_size":block_size, 'tokenizer': tokenizer})
    filtered_vlds = val_data.filter(lambda x: x['keep'])
    print(filtered_vlds)

    val_indices = list(range(len(filtered_vlds)))
        ## distributed sampler
    tr_dist_sampler = DistributedSampler(filtered_tds, shuffle=True,)
    val_dist_sampler = DistributedSampler(filtered_vlds, shuffle= False)

    # Batching the Dataset
    tr_sampler = DistributedWMTBatchSampler(tr_dist_sampler, filtered_tds, max_tokens_per_batch=max_tokens_per_batch)
    val_sampler = DistributedWMTBatchSampler(val_dist_sampler, filtered_vlds, max_tokens_per_batch=max_tokens_per_batch)

    #Load to DataLoader
    train_loader = DataLoader(filtered_tds,
                              batch_sampler=tr_sampler, collate_fn=collate_fn,
                              num_workers=num_workers, pin_memory = True)

    val_loader = DataLoader(filtered_vlds, batch_sampler=val_sampler, 
                            collate_fn=collate_fn, num_workers=num_workers//2, pin_memory=True)

    return train_loader, val_loader, tr_dist_sampler