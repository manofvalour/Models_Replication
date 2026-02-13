import os
import argparse
import time
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch
from torch.distributed import init_process_group, destroy_process_group
from transformers import BertTokenizer
from datasets import load_dataset
from utils import (calculate_bleu, get_transformer_schedule, 
                   save_checkpoint, estimate_loss, data_prep,
                   get_transformer_schedule, load_checkpoint)

from config import TransformerConfig
from transformer_engine import TransformerLanguageModel

def ddp_setup(local_rank):
    init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)

class Trainer:
    def __init__(self, local_rank, rank, world_size, model,
                 optimizer, scheduler, dataset, tokenizer, config):
        
        self.local_rank = local_rank
        self.rank = rank
        self.world_size = world_size
        self.model = model.to(local_rank)
        self.model = DDP(self.model, device_ids=[local_rank])
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.dataset = dataset
        self.tokenizer = tokenizer
        
        if self.local_rank==0:
            os.mkdirs(self.config.ckpt_dir, exist_ok = True)

    def prep_data(self, num_workers=4 ):
        ## cleaning, and preparing and loading the dataset usin data laoder
        train_loader, val_loader = data_prep(self.dataset, self.tokenizer, 
                                             block_size=self.config.block_size, 
                                     max_tokens_per_batch=self.config.max_tokens_per_batch,
                                     num_workers=num_workers)
        
        return train_loader, val_loader
    

    def train(self, train_loader, val_loader,
              start_epoch=0):

        tot_tr_loss, tot_val_loss, tot_norm, tr_ppl, val_ppl = [], [], [], [], []
        running_loss = 0.0
        tot_grand_accum = 0

        for epoch in range(start_epoch, self.config.max_iter):
            self.model.train()
        
            pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")
            val_loss = 0.0

            for batch_idx, (src, tgt) in pbar:
                t0 = time.time()
                src, tgt = src.to(self.config.device), tgt.to(self.config.device)
                tgt_input, tgt_label = tgt[:, :-1], tgt[:, 1:]

                self.optimizer.zero_grad()

                # Mixed Precision
                with torch.autocast(device_type=self.config.device, dtype=torch.bfloat16):
                    logits, loss = self.model(src, tgt_input, tgt_label)

                loss.backward()

                # Stability
                norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                norm
                tot_norm.append(norm.item())

                if batch_idx >0 and batch_idx % 4 == 0:
                    self.optimizer.step()
                    self.scheduler.step()

                    running_loss += loss.item()

                    # Logging every 500 steps
                    if batch_idx > 0 and batch_idx % self.config.train_logging_interval == 0:
                        avg_loss = running_loss / self.config.train_logging_interval
                        avg_ppl = torch.exp(torch.tensor(avg_loss)).item()
                        dt = time.time() - t0
                        toks_per_sec = (src.numel()) / dt
                        tot_tr_loss.append(avg_loss)
                        tr_ppl.append(avg_ppl)      #perplexity

                        avg_loss = torch.Tensor(['avg_loss']).to(self.local_rank)
                        avg_ppl = torch.Tensor(['avg_ppl']).to(self.local_rank)
                        norm = torch.Tensor(['norm']).to(self.local_rank)
                        toks_per_sec = torch.Tensor(['toks_per_sec']).to(self.local_rank)

                        dist.reduce(avg_loss, dst=0, op=dist.ReduceOp.SUM)
                        dist.reduce(avg_ppl, dst=0, op=dist.ReduceOp.SUM)
                        dist.reduce(norm, dst=0, op=dist.ReduceOp.SUM)
                        dist.reduce(toks_per_sec, dst=0, op=dist.ReduceOp.SUM)

                        if self.local_rank == 0:
                            all_gpu_avg_loss = avg_loss / self.world_size
                            all_gpu_avg_ppl = avg_ppl / self.world_size
                            all_gpu_avg_norm = norm / self.world_size
                            all_gpu_toks_per_sec = toks_per_sec

                            pbar.set_postfix({
                                'loss': f"{all_gpu_avg_loss:.2f}",
                                'ppl': f"{all_gpu_avg_ppl:.2f}",
                                'norm': f"{all_gpu_avg_norm:.2f}",
                                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}",
                                'tps': f"{all_gpu_toks_per_sec:.0f}"
                            })
                        running_loss = 0.0 # Reset after logging

                    # Validation Logic (Less frequent to save time)
                    if self.local_rank == 0 and batch_idx > 0 and batch_idx % self.config.eval_interval == 0:
                        self.model.eval()
                        with torch.no_grad():
                            val_loss = estimate_loss(self.model, val_loader, eval_iters=self.config.eval_iter)
                            val_perp = torch.exp(val_loss).item()
                            tot_val_loss.append(val_loss.item())
                            val_ppl.append(val_perp)
                        self.model.train() # Switch back to training

                        tqdm.write(f"Step {batch_idx} | Val Loss: {val_loss.item():.4f} | val_ppl: {val_perp:.2f}")

                    # Get the final validation loss for this epoch (will be the last computed by estimate_loss)
                    final_epoch_val_loss_scalar = val_loss
                    best_model_val_loss = float('inf') # Initialize best validation loss for this epoch

                    if self.local_rank == 0 and final_epoch_val_loss_scalar < best_model_val_loss:
                        best_model_val_loss = final_epoch_val_loss_scalar

                        save_checkpoint(self.model, self.optimizer, self.scheduler, loss.item(), 
                                            final_epoch_val_loss_scalar, epoch, 
                                            path=f"checkpoints/transformer_model_epoch_{epoch}.pt")
                        tqdm.write(f"model saved with Val Loss: {best_model_val_loss:.4f}")


        
def main():
    args = parse_arguements()

    local_rank = int(os.environ.get['LOCAL_RANK', -1])
    world_size = int(os.environ.get['WORLD_SIZE', -1])
    rank = int(os.environ.get['RANK', -1])

    ddp_setup(local_rank=local_rank)

    torch.manual_seed(1337)

    config = TransformerConfig()
    model = TransformerLanguageModel(config,tokenizer=tokenizer).to(local_rank)

    ## loading the dataset and tokenizer
    dataset = load_dataset("sjsurbhi/english-to-french-translation")#)#)                #loaded dataset from HuggingFace "sethjsa/wmt_en_fr_parallel")#
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', use_fast=True)   #loaded tokenizer from Huggingface

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, 
                                betas=(0.9, 0.98), eps=config.eps)

    scheduler = get_transformer_schedule(optimizer, config.n_embd, 
                                        warmup_steps=config.warmup_steps)
    
    trainer = Trainer(local_rank=local_rank, rank=rank, world_size=world_size,
                    model = model, optimizer=optimizer, scheduler=scheduler,
                    dataset = dataset, tokenizer=tokenizer)
    
    train_loader, val_loader = trainer.prep_data(num_workers=8)

    # Training the model
    trainer.train(train_loader=train_loader, val_loader=val_loader)

    destroy_process_group()





if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    main()