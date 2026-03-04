from gpt2_config import GPT2Config
from gpt2_engine import GPT2

import os
import torch
import torch.nn as nn
import torch.nn.function as F
import time
import math
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

num_return_seq = 5
max_entries=30

##======================== SETTING UP THE DEVICES =====================================
device = 'cpu'
if torch.cuda.is_available():
  device = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
  device = 'mps'
print(f"using device: {device}")

## ========================== GETTING DATA BATCHES ====================================
B,T = 16,1024
class DistributedDataLoader:
  def __init__(self, B,T, process_rank, num_of_process):
    import tiktoken
    self.B = B
    self.T = T
    self.process_rank = process_rank
    self.num_of_process = num_of_process

    with open('input.txt', 'r') as f:
        text = f.read()
    
    encoder = tiktoken.get_encoding('gpt2')
    tokens = encoder.encode(text)
    self.tokens = torch.tensor(tokens)

    print(f"Loaded {len(self.tokens)} tokens")
    print(f"Epoch = {len(self.tokens//(B*T))} batches")

    self.current_position = self.B * self.T * self.process_rank

  def next_batch(self):
    B,T = self.B, self.T
    num_of_process = self.num_of_process
    process_rank = self.process_rank

    buf = self.tokens[self.current_position: self.current_position+(B*T)+1]
    x = buf[:-1].view(B,T)
    y = buf[1:].view(B,T)

    self.current_position+= B*T*num_of_process

    if self.current_position + ((B*T*num_of_process)+1) > len(self.tokens):
      self.current_position=B*T*process_rank

    return x,y

##==================== SETTING UP DISTRIBUTED TRAINING ============================
ddp = int(os.environ.get(['RANK', -1]))
if ddp:
  assert torch.cuda.is_available()
  init_process_group(backend='nccl')
  ddp_rank = int(os.environ('RANK'))
  ddp_local_rank = int(os.environ('LOCAL_RANK')) ##multinode settings
  ddp_world_size = int(os.environ('WORLD_SIZE'))
  device = f"cuda:{ddp_local_rank}"
  torch.cuda.set_device(device)
  master_process = ddp_rank==0

else:
  ddp_rank=0
  ddp_local_rank=0
  ddp_world_size=1
  master_process=True
  device = 'cpu'
  if torch.cuda.is_available():
    device = 'cuda'
  elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'
  print(f"Using device: {device}")

##=================================== MODEL TRAINING =======================================
torch.manual_seed(1337)
if torch.cuda.is_available():
  torch.cuda.manual_seed(1337)

total_batch_size = 524288
B=16
T=1024
assert total_batch_size //(B*T*ddp_world_size) ==0 , "make sure total batch size is divisible by B*T"
grad_accum_step = total_batch_size//(B*T*ddp_world_size) ## total_batch_soze//B*T*ddp_world_size
if master_process:
  print(f"total desired batch size: {total_batch_size}")
  print(f"==> Calculated gradient Accum step: {grad_accum_step}")


train_loader = DistributedDataLoader(B=4,T=32, process_rank = ddp_rank, 
                              num_processes=ddp_world_size)

torch.set_float32_matmul_precision('high') #optim3 (for gpu to use tf32 for matmul ops) 
model = GPT2(GPT2Config())
#model = DDP(model)
model.to(device)
model = torch.compile(model) #optim1 (kernel fusion)
if ddp:
  model = DDP(model, device_ids =[ddp_local_rank])
raw_model = model.module if ddp else model

##================== SETTING UP LR ==================
max_lr = 6e-4
min_lr = max_lr *0.1
warmup_steps = 10
max_step=50

def get_lr(it):
  if it< warmup_steps:
    return max_lr * (it+1)/warmup_steps
  if it > max_step:
    return min_lr
  
  decay_ratio = (it - warmup_steps) / (max_step - warmup_steps)
  assert 0<= decay_ratio <=1
  coef = 0.5 *(1.0 + math.cos(math.pi * decay_ratio))
  return min_lr + coef * (max_lr - min_lr)


##================ TRAINING LOOP ======================
optimizer = raw_model.configure_optimizers(weight_decay = 0.1, lr = 6e-4, device = device)
for step in range(max_step):
  t0 = time.time()
  optimizer.zero_grad()
  loss_accum = 0.0
  for micro_step in range(grad_accum_step):

    x,y = train_loader.next_batch()
    x,y = x.to(device), y.to(device)

    with torch.autocast(device_type=device, dtype=torch.bfloat16): #optim2 (mixed precision)
      logits, loss = model(x,y)
    loss = loss/grad_accum_step
    loss_accum +=loss.detach()
    if ddp:
      model.require_backward_grad_sync = (micro_step == grad_accum_step-1)
    loss.backward()
  if ddp:
    dist.all_reduce(loss_accum, ops = dist.ReduceOp.AVG)

  norm=torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
  lr = get_lr(step)
  for param_group in optimizer.param_groups:
    param_group['lr']= lr
  
  optimizer.step()
  torch.cuda.synchronize()

  t1 = time.time()
  dt = (t1-t0) #*1000 #ms
  tokens_processed = train_loader.B * train_loader.T * grad_accum_step* ddp_world_size
  toks_per_sec = tokens_processed/dt

  if master_process:
    print(f"step: {step} | loss: {loss_accum.item():.2f} | norm: {norm:.4f} | lr: {lr:.4f} | dt: {dt:.2f}ms | toks/sec: {toks_per_sec}")

if ddp:
  destroy_process_group()

import sys; sys.exit()

############# MODEL GENERATION ################
#model = GPT2.pretrained('gpt2')
model = GPT2(GPT2Config())
model.eval()
model.to(device)

# prefill phase
import tiktoken
enc= tiktoken.get_encoding('gpt2')
tokens= enc.encode("Hello, I'm a Language model,")
tokens = torch.tensor(tokens, dtype = torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_seq, 1)
x = tokens.to(device)
print(x.shape)

## the prediction phase
torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1)< max_entries:
  with torch.no_grad():
    logits, _ = model(x)
    logits = logits[:, -1, :]
    probs = F.softmax(logits, dim=-1)
    topk_probs, topk_indices = torch.topk(probs, k=50,dim=-1)
    ix = torch.multinomial(topk_probs, 1)
    xcol = torch.gather(topk_indices, -1, ix)
    x = torch.cat((x, xcol), dim=1)

for i in range(num_return_seq):
  tokens = x[i, :max_entries].tolist()
  dec = enc.decode(tokens)
  print(">", dec)



