import os
import torch
import wandb
import time
import sacrebleu
from tqdm import tqdm
from transformers import T5Tokenizer
from transformers import BertTokenizer
from datasets import load_dataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from contextlib import nullcontext
from transformers import get_linear_schedule_with_warmup

from transformer_engine import Transformer
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
              start_iter=0):

        ddp = int(os.environ.get('LOCAL_RANK', -1)) != -1
        best_model_val_loss = float('inf')
        val_loss = torch.tensor(float('inf'), device=self.device)
        os.environ["TORCH_CUDAGRAPHS_EAGER_FALLBACK"] = "1"
        torch._dynamo.config.optimize_ddp = True
        torch.set_float32_matmul_precision('high')
    
        if ddp:
            assert torch.cuda.is_available(), "Distributed training requires CUDA"
            ddp_rank = int(os.environ['RANK'])
            ddp_local_rank = int(os.environ['LOCAL_RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            
            master_process = ddp_rank == 0
            print(f"Initialized distributed training on rank {ddp_rank}; (local rank : {ddp_local_rank}) with world size {world_size}")

        else:
            ddp_rank = 0
            ddp_local_rank = 0
            world_size = 1
            master_process = True
            print(f"Single-process mode on {self.device}")


        model = self.model.to(self.device)
        if ddp:
            model= DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=True)

        model = torch.compile(model)        
        raw_model = model.module if ddp else model

        total_batches = self.config.total_batches
        mini_batch = self.config.batch_size * self.config.block_size
        assert total_batches % (mini_batch * world_size) == 0, "total_batches must be divisible by mini_batch"
        accum_steps = total_batches//(mini_batch * world_size)
        
        if master_process:
            print('total_batches:', total_batches, 'mini_batch:', mini_batch, 'accum_steps:', accum_steps)

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        scaler = torch.amp.GradScaler('cuda')
        global_step = 0

        #for epoch in range(start_epoch, self.config.max_iter):
        model.train()
        #dist_sampler.set_epoch(epoch)
        #pbar = tqdm(enumerate(train_loader), total=self.config.max_iter))
        train_iter = iter(train_loader)
        running_loss = torch.zeros(1, device = self.device)
        train_accum_count = 0

        for iteration in range(start_iter, self.config.max_iter):
        #for batch_idx, (src, tgt) in pbar:
            t0 = time.time()

            try:
                src, tgt = next(train_iter)
            except StopIteration:
                dist_sampler.set_epoch(iteration)  # new epoch
                train_iter = iter(train_loader)
                src, tgt = next(train_iter)
          
            src, tgt = src.to(self.device), tgt.to(self.device)
            tgt_input, tgt_label = tgt[:, :-1], tgt[:, 1:]

            # Mixed Precision
            with torch.autocast(device_type=self.config.device, dtype=dtype):
                loss = model(src, tgt_input, tgt_label)

            loss = loss/accum_steps
            running_loss += loss.detach()
            #print(running_loss)

            context = model.no_sync() if ddp and (global_step + 1) % accum_steps != 0 else nullcontext()
            with context:
                if dtype == torch.bfloat16:
                    loss.backward()
                else:
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.optimizer)

            if (global_step+1) % accum_steps == 0:

                if ddp:
                    dist.all_reduce(running_loss, op=dist.ReduceOp.AVG)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)

                if dtype == torch.bfloat16:
                    self.optimizer.step()
                    self.scheduler.step()
                else:
                    scaler.step(self.optimizer)
                    self.scheduler.step()
                    scaler.update()

                self.optimizer.zero_grad(set_to_none =True)
                train_accum_count+=1

                # Logging every 500 steps
                if global_step !=0 and (global_step+1) % (self.config.train_logging_interval) == 0:
                    
                    avg_loss = running_loss / train_accum_count
                    avg_ppl = torch.exp(avg_loss).item()
                    torch.cuda.synchronize() # Ensure all GPU computations are done before timing
                    dt = (time.time() - t0)
                    toks_per_sec = (src.numel() * accum_steps* world_size) / dt
                    running_loss = torch.zeros(1, device = self.device)

                    if master_process:
                        wandb.log({
                            "train/loss": avg_loss,
                            "train/ppl": avg_ppl,
                            "train/lr": self.scheduler.get_last_lr()[0],
                            "train/grad_norm": norm,

                        }, step=global_step)
                        train_accum_count=0

                        tqdm.write(f"step: {iteration+1} | train_loss: {avg_loss.item():.2f} | ppl: {avg_ppl:.2f} | norm: {norm:.2f} | tps: {toks_per_sec:.0f} | dt: {dt*accum_steps:.0f}secs | lr: {self.scheduler.get_last_lr()[0]:.2e}")
                        
                    #running_loss.zero_() # Reset after logging

                # Validation Logic (Less frequent to save time)
                if (global_step+1) % (self.config.eval_interval) == 0:
                    
                    val_loss = estimate_loss(model, val_loader, eval_iters=self.config.eval_iter, device = self.device)
                    val_perp = torch.exp(val_loss).item()
                        
                    if ddp_local_rank == 0:
                        wandb.log({
                            "val/loss": val_loss,
                            "val/ppl": val_perp
                        }, step = global_step)
                        tqdm.write(f"Step {global_step+1} | Val Loss: {val_loss.item():.4f} | val_ppl: {val_perp:.2f}")
                    
                        # Get the final validation loss for this epoch (will be the last computed by estimate_loss)
                        final_epoch_val_loss_scalar = val_loss

                        if final_epoch_val_loss_scalar < best_model_val_loss:
                            best_model_val_loss = final_epoch_val_loss_scalar

                            save_checkpoint(raw_model, self.optimizer, self.scheduler, loss.item(), 
                                                final_epoch_val_loss_scalar, iteration, 
                                                path=f"checkpoints/best_model.pt")
                            print(f"model saved with Val Loss: {best_model_val_loss:.4f}")
                        
                        
                # generating response and calculating bleu score every 1000 steps
                if master_process:
                    if (global_step+1) % (self.config.generate_interval) == 0:  # add this config field
                        model.eval()
                        with torch.no_grad():
                            # Generate translations for a batch of validation examples
                            val_iter = iter(val_loader)

                            try:
                                src, tgt = next(val_iter)
                            except StopIteration:
                                dist_sampler.set_epoch(iteration)  # new epoch
                                train_iter = iter(val_iter)
                                src, tgt = next(val_iter)
                        
                            src_batch, tgt_batch = src.to(self.device), tgt.to(self.device)
                            generated_ids = raw_model.generate(src_batch, max_len=50)
                            
                            # Decode generated and reference translations
                            generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                            reference_texts = self.tokenizer.batch_decode(tgt_batch, skip_special_tokens=True)

                            ##printing the source and the generated translation for the first example in the batch
                            print(f"Source: {self.tokenizer.decode(src_batch[0], skip_special_tokens=True)}")
                            print(f"Generated: {generated_texts[0]}")
                            print(f"Reference: {reference_texts[0]}")

                            # Calculate BLEU score
                            bleu_score = sacrebleu.corpus_bleu(generated_texts, [reference_texts])
                            tqdm.write(f"Step {global_step} | BLEU Score: {bleu_score.score:.2f}")
                        model.train() # Switch back to training

                if ddp_local_rank ==0:
                        # always save latest for crash recovery
                        save_checkpoint(raw_model, self.optimizer, self.scheduler, loss.item(),
                                       val_loss, iteration,
                                      path="checkpoints/latest.pt")
            global_step+=1
            if ddp:
                dist.barrier(device_ids=[ddp_local_rank])
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
    # Option 1: Use T5 tokenizer (better for translation)
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    #tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', use_fast=True)   #loaded tokenizer from Huggingface
    config = TransformerConfig()

    wandb.init(
        project="vanilla_transformer-en-fr",
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
     # ── RESUME LOGIC ───────────────────────────────────────────────────
    start_step = 0
    resume_path = "checkpoints/latest.pt"


    model = Transformer(config, vocab_size=tokenizer.vocab_size, pad_token_id=tokenizer.pad_token_id)
    tot_params = sum(p.numel() for p in model.parameters())  
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) 

    print(f"Number of parameters: {tot_params/1e6}M")
    print(f"Number of trainable parameters: {trainable_params/1e6}M")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, 
                                betas=(0.9, 0.98), eps=config.eps)
    total_steps = config.max_iter * (config.total_batches/(config.batch_size * config.block_size))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=config.warmup_steps,
        num_training_steps = total_steps

    )
   # scheduler = get_transformer_schedule(optimizer, config.n_embd, 
    #                                    warmup_steps=config.warmup_steps)

    if os.path.exists(resume_path):
        model, optimizer, scheduler, start_step, train_loss, val_loss = load_checkpoint(
            model, optimizer, scheduler, resume_path, device
        )
        print(f"Resuming from step {start_step}, train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
    else:
        print("No checkpoint found, starting from scratch")

    model = model.to(device)
    
    trainer = Trainer(model = model, optimizer=optimizer, scheduler=scheduler,
                    dataset = dataset, tokenizer=tokenizer, device = device, config = config)
    
    train_loader, val_loader,dist_sampler = trainer.prep_data(num_workers=8)

    # Training the model
    trainer.train(train_loader=train_loader, val_loader=val_loader, dist_sampler = dist_sampler, start_iter=start_step)


if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    main()