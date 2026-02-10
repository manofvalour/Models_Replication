import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from datasets import load_dataset
from utils import (calculate_bleu, get_transformer_schedule, 
                   save_checkpoint, estimate_loss, data_prep,
                   get_transformer_schedule, load_checkpoint)
import time
from tqdm import tqdm
import os

from config import TransformerConfig
from transformer_engine import TransformerLanguageModel

torch.manual_seed(1337)

config = TransformerConfig()

## loading the dataset and tokenizer
dataset = load_dataset("sjsurbhi/english-to-french-translation")#)#)                #loaded dataset from HuggingFace "sethjsa/wmt_en_fr_parallel")#
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', use_fast=True)   #loaded tokenizer from Huggingface

## cleaning, and preparing and loading the dataset usin data laoder
train_loader, val_loader = data_prep(dataset, tokenizer, block_size=config.block_size, 
                                     max_tokens_per_batch=config.block_size * config.batch_size,
                                     num_workers=4)

## initializing the model
model = TransformerLanguageModel(config,tokenizer=tokenizer).to(config.device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, 
                             betas=(0.9, 0.98), eps=config.eps)

scheduler = get_transformer_schedule(optimizer, config.n_embd, 
                                     warmup_steps=config.warmup_steps)

def train(model, optimizer, scheduler, train_loader, val_loader, 
          max_iter, eval_iter, eval_interval, train_logging_interval, 
          start_epoch=0,device='cpu'):

    tot_tr_loss, tot_val_loss, tot_norm, tr_ppl, val_ppl = [], [], [], [], []
    running_loss = 0.0
    total_steps = len(train_loader) * 3

    for epoch in range(start_epoch, max_iter):
        model.train()
    
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")
        val_loss = 0.0

        for batch_idx, (src, tgt) in pbar:
            t0 = time.time()
            src, tgt = src.to(device), tgt.to(device)
            tgt_input, tgt_label = tgt[:, :-1], tgt[:, 1:]

            optimizer.zero_grad()

            # Mixed Precision
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(src, tgt_input, tgt_label)

            loss.backward()

            # Stability
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            norm
            tot_norm.append(norm.item())

            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

            # Logging every 500 steps
            if batch_idx > 0 and batch_idx % train_logging_interval == 0:
                avg_loss = running_loss / train_logging_interval
                avg_ppl = torch.exp(torch.tensor(avg_loss)).item()
                dt = time.time() - t0
                toks_per_sec = (src.numel()) / dt
                tot_tr_loss.append(avg_loss)
                tr_ppl.append(avg_ppl)      #perplexity

                pbar.set_postfix({
                    'loss': f"{avg_loss:.2f}",
                    'ppl': f"{avg_ppl:.2f}",
                    'norm': f"{norm:.2f}",
                    'lr': f"{scheduler.get_last_lr()[0]:.2e}",
                    'tps': f"{toks_per_sec:.0f}"
                })
                running_loss = 0.0 # Reset after logging

            # Validation Logic (Less frequent to save time)
            if batch_idx > 0 and batch_idx % eval_interval == 0:
                model.eval()
                with torch.no_grad():
                    val_loss = estimate_loss(model, val_loader, eval_iters=eval_iter)
                    val_perp = torch.exp(val_loss).item()
                    tot_val_loss.append(val_loss.item())
                    val_ppl.append(val_perp)
                model.train() # Switch back to training
                tqdm.write(f"Step {batch_idx} | Val Loss: {val_loss.item():.4f} | val_ppl: {val_perp:.2f}")

          # Get the final validation loss for this epoch (will be the last computed by estimate_loss)
        final_epoch_val_loss_scalar = val_loss
        best_model_val_loss = float('inf') # Initialize best validation loss for this epoch

        if final_epoch_val_loss_scalar < best_model_val_loss:
            best_model_val_loss = final_epoch_val_loss_scalar

            save_checkpoint(model, optimizer, scheduler, loss.item(), 
                                final_epoch_val_loss_scalar, epoch, 
                                path=f"checkpoints/transformer_model_epoch_{epoch}.pt")
            tqdm.write(f"model saved with Val Loss: {best_model_val_loss:.4f}")
            

if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    train(model, optimizer, scheduler, train_loader, val_loader, 
          max_iter=config.max_iter, eval_iter=config.eval_iter, 
          eval_interval=config.eval_interval, 
          train_logging_interval=config.train_logging_interval,
          start_epoch=0, device=config.device)