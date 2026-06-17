from gpt2_config import GPT2Config
from gpt2_engine import GPT2
from gpt_utils import DistributedDataLoader, evaluate, get_lr, load_text
from gpt2_config import GPT2Config

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

def distributed_training(config, data, rank, 
                         local_rank, world_size, 
                         device):

  ddp = int(os.environ.get(['RANK', -1]))
  master_process=rank==0

  #setting up the distributed training run
  torch.manual_seed(1337)
  if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

  total_batch_size = config.TOT_BATCH_SIZE
  B = config.BATCH_SIZE
  T = config.SEQ_LEN

  assert total_batch_size //(B*T*world_size) ==0 , "make sure total batch size is divisible by B*T"
  grad_accum_step = total_batch_size//(B*T*world_size) ## total_batch_soze//B*T*ddp_world_size
  
  if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"==> Calculated gradient Accum step: {grad_accum_step}")

  ##getting the data
  distributed_dataloader = DistributedDataLoader(config.BATCH_SIZE, 
                                                config.SEQ_LEN,rank, 
                                                world_size)
  train_data, val_data = distributed_dataloader.encode_and_split(data, n=0.8)


  torch.set_float32_matmul_precision('high') #optim3 (for gpu to use tf32 for matmul ops) 
  model = GPT2(GPT2Config())
  model.to(device)
  model = torch.compile(model) #optim1 (kernel fusion)
  
  if ddp:
    model = DDP(model, device_ids =[local_rank])
  raw_model = model.module if ddp else model

  ##training loop
  optimizer = raw_model.configure_optimizers(weight_decay = 0.1, 
                                             lr = 6e-4, device = device)
  for step in range(config.MAX_STEPS):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_step):

      x,y = distributed_dataloader.next_batch(train_data)
      x,y = x.to(device), y.to(device)

      with torch.autocast(device_type=device, dtype=torch.bfloat16): #optim2 (mixed precision)
        _, loss = model(x,y)

      loss = loss/grad_accum_step
      loss_accum +=loss.detach()

      if ddp:
        model.require_backward_grad_sync = (micro_step == grad_accum_step-1)
      loss.backward()

    if ddp:
      dist.all_reduce(loss_accum, ops = dist.ReduceOp.AVG)

    norm=torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    lr = get_lr(step, config)
    for param_group in optimizer.param_groups:
      param_group['lr']= lr
    
    optimizer.step()
    torch.cuda.synchronize()

    t1 = time.time()
    dt = (t1-t0) #*1000 #ms
    tokens_processed = distributed_dataloader.B * distributed_dataloader.T * grad_accum_step* world_size
    toks_per_sec = tokens_processed/dt
    train_perp = math.exp(loss_accum)

    if master_process:
      if (step+1) % config.TRAIN_STEP==0:
        print(f"step: {step} | loss: {loss_accum.item():.2f} | train_perp: {train_perp:.2f} | norm: {norm:.4f} | lr: {lr:.4f} | dt: {dt:.2f}ms | toks/sec: {toks_per_sec}")

      if (step+1)%config.EVAL_STEP==0:
        val_loss = evaluate(model, val_data, 
                          config.BATCH_SIZE, 
                          config.SEQ_LEN, 
                          device)
        val_perp = math.exp(val_loss)
        print(f"step: {step+1} | val_loss: {val_loss:.2f} | val_perp: {val_perp:.2f}")


      if (step+1)%config.GEN_EVAL ==0:
        #model = GPT2.pretrained('gpt2')
        model.eval()
        
        #the prefill phase
        num_return_seq = 3
        text= "Hello, I'm a Language model,"
        tokens = distributed_dataloader.encode_and_split(text, n=0)
        tokens = tokens.unsqueeze(0).repeat(num_return_seq, 1)

        x = tokens.to(device)
        print(x.shape)

        ##the prediction phase
        x = model.generate(x, config.MAX_NEW_TOKENS)

        for i in range(num_return_seq):
          tokens = x[i, :config.MAX_NEW_TOKENS].tolist()
          dec = distributed_dataloader.decode(tokens)
          print(">>>", dec)
  if ddp:
    destroy_process_group()

#import sys; sys.exit()


def main():

  #setting up the distributed training run
  ddp = int(os.environ.get(['RANK', -1]))
  if ddp:
    assert torch.cuda.is_available()
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ('RANK'))
    ddp_local_rank = int(os.environ('LOCAL_RANK')) ##multinode settings
    ddp_world_size = int(os.environ('WORLD_SIZE'))
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)

  else:
    ddp_rank=0
    ddp_local_rank=0
    ddp_world_size=1

    device = 'cpu'
    if torch.cuda.is_available():
      device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
      device = 'mps'
    print(f"Using device: {device}")


  data = load_text("GPT2/wizard_of_oz.txt")
  config= GPT2Config()
  distributed_training(config, data, ddp_rank, ddp_local_rank, ddp_world_size, device)

if __name__ == "__main__":
  main()