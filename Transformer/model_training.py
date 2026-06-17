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
import torch.multiprocessing as mp

from transformer_engine import Transformer
from utils import (calculate_bleu, get_transformer_schedule,
                   save_checkpoint, estimate_loss, data_prep,
                   get_transformer_schedule, load_checkpoint)
from config import TransformerConfig
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", message=".*has no effect in the context.*")

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
              start_iter=0, max_iter=5, experiment_id:str = None):

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
            model= DDP(model, device_ids=[ddp_local_rank])

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

        for iteration in range(start_iter, max_iter):
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

            aloss = loss/accum_steps
            running_loss += aloss.detach()
            #print(running_loss)

            context = model.no_sync() if ddp and (global_step + 1) % accum_steps != 0 else nullcontext()
            with context:
                if dtype == torch.bfloat16:
                    aloss.backward()
                else:
                    scaler.scale(aloss).backward()
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
                if (global_step + 1) % self.config.train_logging_interval == 0 or \
                   (global_step + 1) % self.config.eval_interval == 0:
                
                    log_dict = {}
                    log_step = iteration + 1
                
                    if (global_step + 1) % self.config.train_logging_interval == 0 and global_step != 0:
                        avg_loss = running_loss / train_accum_count
                        avg_ppl = torch.exp(avg_loss).item()
                        torch.cuda.synchronize()
                        dt = (time.time() - t0)
                        toks_per_sec = (src.numel() * accum_steps * world_size * self.config.train_logging_interval) / dt
                        last_log_time = time.time()
                        running_loss = torch.zeros(1, device=self.device)
                
                        log_dict.update({
                            "train/loss": avg_loss,
                            "train/ppl": avg_ppl,
                            "train/lr": self.scheduler.get_last_lr()[0],
                            "train/grad_norm": norm,
                        })
                        if master_process:
                            train_accum_count = 0
                            tqdm.write(f"step: {iteration+1} | train_loss: {avg_loss.item():.2f} | ppl: {avg_ppl:.2f} | norm: {norm:.2f} | tps: {toks_per_sec:.0f} | dt: {dt:.2f}secs | lr: {self.scheduler.get_last_lr()[0]:.2e}")
                
                    if (global_step + 1) % self.config.eval_interval == 0:
                        if master_process:
                            val_loss = estimate_loss(model, val_loader, eval_iters=self.config.eval_iter, device=self.device)
                            model.train()  # restore here, not inside estimate_loss
                            
                            val_perp = torch.exp(val_loss).item()
                            log_dict.update({
                                "val/loss": val_loss,
                                "val/ppl": val_perp,
                            })
                            tqdm.write(f"Step: {iteration+1} | Val Loss: {val_loss.item():.4f} | val_ppl: {val_perp:.2f}")
                
                            if val_loss < best_model_val_loss:
                                best_model_val_loss = val_loss.item()
                                save_checkpoint(raw_model, self.optimizer, self.scheduler, avg_loss.item(),
                                                val_loss, iteration,
                                                path="checkpoints/best_model.pt",
                                                wandb_run_id=experiment_id)
                                print(f"model saved with Val Loss: {best_model_val_loss.item():.4f}")
                
                    if master_process and log_dict:
                        wandb.log(log_dict, step=log_step)            
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
                            generated_ids = raw_model.generate(src_batch, max_len=150)
                            
                            # Decode generated and reference translations
                            generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                            reference_texts = self.tokenizer.batch_decode(tgt_batch, skip_special_tokens=True)

                            ##printing the source and the generated translation for the first example in the batch
                            print(f"Source: {self.tokenizer.decode(src_batch[0], skip_special_tokens=True)}")
                            print(f"Generated: {generated_texts[0]}")
                            print(f"Reference: {reference_texts[0]}")

                            # Calculate BLEU score
                            bleu_score = sacrebleu.corpus_bleu(generated_texts, [reference_texts])
                            tqdm.write(f"Step {global_step+1} | BLEU Score: {bleu_score.score:.2f}")
                        model.train() # Switch back to training

                if ddp_local_rank ==0:
                        # always save latest for crash recovery
                        save_checkpoint(raw_model, self.optimizer, self.scheduler, loss.item(),
                                       val_loss, iteration, wandb_run_id=experiment_id,
                                      path="checkpoints/latest.pt",)
            global_step=iteration+1
            if ddp:
                dist.barrier(device_ids=[ddp_local_rank])
        if ddp:
            destroy_process_group()
            
import traceback   
def main(rank, world_size,
         model_location='None', 
         resume_training=False):
    try:
        os.environ['RANK'] = str(rank)
        os.environ['LOCAL_RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        os.environ['NCCL_P2P_DISABLE'] = '1'
        
        ddp = world_size > 1
        
        if ddp:
            init_process_group(backend='nccl', rank =rank, world_size= world_size)
            torch.cuda.set_device(int(rank))
        
        torch.manual_seed(1337)
        device = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'
    
        ## loading the dataset and tokenizer
        dataset = load_dataset("sjsurbhi/english-to-french-translation")#)#) #loaded dataset from HuggingFace "sethjsa/wmt_en_fr_parallel")#
        tokenizer = T5Tokenizer.from_pretrained('t5-base', lagacy=False)
        #tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', use_fast=True)   #loaded tokenizer from Huggingface
        config = TransformerConfig()

        master_process = rank ==0
    
         # ── RESUME LOGIC ───────────────────────────────────────────────────
        start_step = 0
        resume_path = model_location
    
    
        model = Transformer(config, vocab_size=tokenizer.vocab_size, pad_token_id=tokenizer.pad_token_id)
        tot_params = sum(p.numel() for p in model.parameters())  
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) 
    
        if master_process:
            print(f"Number of parameters: {tot_params/1e6}M")
            print(f"Number of trainable parameters: {trainable_params/1e6}M")
            
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, 
                                    betas=(0.9, 0.98), eps=config.eps)
        total_steps = config.max_iter
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=config.warmup_steps,
            num_training_steps = total_steps
    
        )
       # scheduler = get_transformer_schedule(optimizer, config.n_embd, 
        #                                    warmup_steps=config.warmup_steps)
    
        if os.path.exists(model_location) and resume_training:
            model, optimizer, scheduler, start_step, train_loss, val_loss, wandb_run_id = load_checkpoint(
                model, optimizer, scheduler, model_location, device
            )
            if master_process:
                wandb.init(
                    project="vanilla_transformer-en-fr",
                    entity="ajalae2-emmanuel-ajala",
                    id = wandb_run_id,
                    resume = "must",
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
                print(f"Resuming from step {start_step}, train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        else:
            if master_process:
                wandb.init(
                project="vanilla_transformer-en-fr",
                entity="ajalae2-emmanuel-ajala",
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
                print("No checkpoint found, starting from scratch")
    
        model = model.to(device)
        
        trainer = Trainer(model = model, optimizer=optimizer, scheduler=scheduler,
                        dataset = dataset, tokenizer=tokenizer, 
                        device = device, config = config)
        
        train_loader, val_loader,dist_sampler = trainer.prep_data(num_workers=4)
    
        # Training the model
        trainer.train(train_loader=train_loader, 
                      val_loader=val_loader, 
                      dist_sampler = dist_sampler, 
                      start_iter=start_step, 
                      max_iter=config.max_iter,
                     experiment_id =wandb.init().id )

    except Exception:
        traceback.print_exc()
        raise

if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    world_size = torch.cuda.device_count()
    if world_size>1:
        mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
  
    else:
        main(0,1)