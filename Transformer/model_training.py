import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from datasets import load_dataset
from tqdm import tqdm
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import wandb
from contextlib import nullcontext
import time
from torch.utils.data import DataLoader, DistributedSampler
from transformer_engine import Transformer
from sacrebleu

from utils import (calculate_bleu, get_transformer_schedule,
                   save_checkpoint, estimate_loss, data_prep,
                   get_transformer_schedule, load_checkpoint)
from config import TransformerConfig


class Trainer:
    def __init__(self, model, device, optimizer, 
                 scheduler, dataset, tokenizer, config):
        
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device= device

    def prep_data(self, num_workers=4):
        ## cleaning, and preparing and loading the dataset usin data laoder
        train_loader, val_loader, dist_sampler = data_prep(self.dataset, self.tokenizer, 
                                             block_size=self.config.block_size, 
                                     max_tokens_per_batch=self.config.max_tokens_per_batch,
                                     num_workers=num_workers)
        
        return train_loader, val_loader, dist_sampler
    

    def train(self, train_loader, val_loader, dist_sampler,
              start_epoch=0):

        ddp = int(os.environ.get('LOCAL_RANK', -1)) != -1
        best_model_val_loss = float('inf') # Initialize best validation loss for this epoch
        val_loss = torch.tensor(float('inf'), device=self.device)
        os.environ["TORCH_CUDAGRAPHS_EAGER_FALLBACK"] = "1"
        torch._dynamo.config.optimize_ddp = True
        torch.set_float32_matmul_precision('high')
    
        if ddp:
            assert torch.cuda.is_available(), "Distributed training requires CUDA"
            #init_process_group(backend='nccl')
            ddp_rank = int(os.environ['RANK'])
            ddp_local_rank = int(os.environ['LOCAL_RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            master_process = ddp_rank == 0
            print(f"Initialized distributed training on rank {ddp_rank} (local rank {ddp_local_rank}) with world size {world_size}")

        else:
            ddp_rank = 0
            ddp_local_rank = 0
            world_size = 1
            master_process = True
            print(f"Single-process mode on {self.device}")
        
        model = self.model.to(self.device)
        model = torch.compile(model)

        if ddp:
            model= DDP(model, device_ids=[ddp_local_rank])
        
        raw_model = model.module if ddp else model

        total_batches = 32768
        mini_batch = 4096
        assert total_batches % mini_batch * world_size == 0, "total_batches must be divisible by mini_batch"
        accum_steps = total_batches//(mini_batch * world_size)
        
        if master_process:
            print('total_batches:', total_batches, 'mini_batch:', mini_batch, 'accum_steps:', accum_steps)

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        global_step = 0

        for epoch in range(start_epoch, self.config.max_iter):
            model.train()
            dist_sampler.set_epoch(epoch)
            pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")
            
            running_loss = torch.zeros(1, device = self.device)
            #t0 = time.time()

            for batch_idx, (src, tgt) in pbar:
                t0 = time.time()
                src, tgt = src.to(self.config.device), tgt.to(self.config.device)
                tgt_input, tgt_label = tgt[:, :-1], tgt[:, 1:]

                # Mixed Precision
                with torch.autocast(device_type=self.config.device, dtype=dtype):
                    logits, loss = model(src, tgt_input, tgt_label)

                loss = loss/accum_steps
                running_loss += loss.detach()

                context = model.no_sync() if ddp and (batch_idx + 1) % accum_steps != 0 else nullcontext()
                with context:
                    loss.backward()

                if (batch_idx+1) % accum_steps == 0:
                    if ddp:
                        dist.all_reduce(running_loss, op=dist.ReduceOp.AVG)
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none =True)
                    global_step += 1

                    # Logging every 500 steps
                    if global_step % self.config.train_logging_interval == 0:
                        
                        avg_loss = running_loss / self.config.train_logging_interval
                        avg_ppl = torch.exp(avg_loss).item()
                        torch.cuda.synchronize() # Ensure all GPU computations are done before timing
                        dt = (time.time() - t0) * accum_steps
                        toks_per_sec = (src.numel() * accum_steps* world_size) / dt

                        if master_process:
                            wandb.log({
                                "train/loss": avg_loss,
                                "train/ppl": avg_ppl,
                                "train/lr": self.scheduler.get_last_lr()[0],
                                "train/grad_norm": norm,

                            }, step=global_step)
                            pbar.set_postfix({
                                'loss': f"{avg_loss.item():.2f}",
                                'ppl': f"{avg_ppl:.2f}",
                                'norm': f"{norm:.2f}",
                                'tps': f"{toks_per_sec:.0f}",
                                'dt': f"{dt:.0f}secs",
                                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}",
                            })
                        running_loss.zero_() # Reset after logging

                    # Validation Logic (Less frequent to save time)
                    if global_step % self.config.eval_interval == 0:
                        
                        val_loss = estimate_loss(model, val_loader, eval_iters=self.config.eval_iter, device = self.device)
                        val_perp = torch.exp(val_loss).item()
                            
                        if ddp_local_rank == 0:
                            wandb.log({
                                "val/loss": val_loss,
                                "val/ppl": val_perp
                            }, step = global_step)
                            tqdm.write(f"Step {global_step} | Val Loss: {val_loss.item():.4f} | val_ppl: {val_perp:.2f}")
                        
                        
                    ## generating response and calculating bleu score every 1000 steps
            #if ddp_local_rank == 0 and global_step % 20 == 0: #self.config.eval_interval == 0:
            model.eval()
            with torch.no_grad():
                # Generate translations for a batch of validation examples
                src_batch, tgt_batch = next(iter(val_loader))
                src_batch, tgt_batch = src_batch.to(self.device), tgt_batch.to(self.device)
                generated_ids = raw_model.generate(src_batch, max_len=50)
                
                # Decode generated and reference translations
                generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                reference_texts = self.tokenizer.batch_decode(tgt_batch, skip_special_tokens=True)

                ##printing the source and the generated translation for the first example in the batch
                print(f"Source: {self.tokenizer.decode(src_batch[0], skip_special_tokens=True)}")
                print(f"Generated: {generated_texts[0]}")
                print(f"Reference: {reference_texts[0]}")

                # Calculate BLEU score
                #bleu_score = calculate_bleu(self.tokenizer, generated_texts, reference_texts)
                bleu_score = sacrebleu.corpus_bleu(generated_texts, [reference_texts])
                tqdm.write(f"Step {global_step} | BLEU Score: {bleu_score.score:.2f}")
            model.train() # Switch back to training

            # Get the final validation loss for this epoch (will be the last computed by estimate_loss)
            final_epoch_val_loss_scalar = val_loss

            if ddp_local_rank == 0 and final_epoch_val_loss_scalar < best_model_val_loss:
                best_model_val_loss = final_epoch_val_loss_scalar

                save_checkpoint(raw_model, self.optimizer, self.scheduler, loss.item(), 
                                    final_epoch_val_loss_scalar, epoch, 
                                    path=f"checkpoints/transformer_model_epoch_{epoch}.pt")
                tqdm.write(f"model saved with Val Loss: {best_model_val_loss:.4f}")

            if ddp:
                dist.barrier()
        if ddp:
            destroy_process_group()
        
def main():
    
    ddp = int(os.environ.get('LOCAL_RANK', -1)) != -1
    
    if ddp:
        init_process_group(backend='nccl')
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

    torch.manual_seed(1337)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    
    ## loading the dataset and tokenizer
    dataset = load_dataset("sjsurbhi/english-to-french-translation")#)#)                #loaded dataset from HuggingFace "sethjsa/wmt_en_fr_parallel")#
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', use_fast=True)   #loaded tokenizer from Huggingface

    config = TransformerConfig()
    wandb.init(
        project="transformer-en-fr",
        entity="ajalae2-emmanuel-ajala", # Your specific entity
        config={
            "learning_rate": config.learning_rate,
            "n_embd": config.n_embd,
            "n_head": config.n_head,
            "n_layer": config.n_layer,
            "block_size": config.block_size,
            "warmup_steps": config.warmup_steps,
            "device": config.device,
            'batch_size': config.max_tokens_per_batch * 8
        }
    )

    model = Transformer(config, vocab_size=tokenizer.vocab_size, pad_token_id=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, 
                                betas=(0.9, 0.98), eps=config.eps)

    scheduler = get_transformer_schedule(optimizer, config.n_embd, 
                                        warmup_steps=config.warmup_steps)

    trainer = Trainer(model = model, optimizer=optimizer, scheduler=scheduler,
                    dataset = dataset, tokenizer=tokenizer, device = device, config = config)
    
    train_loader, val_loader,dist_sampler = trainer.prep_data(num_workers=8)

    # Training the model
    trainer.train(train_loader=train_loader, val_loader=val_loader, dist_sampler = dist_sampler)


if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    main()