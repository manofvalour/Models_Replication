from gpt2_config import GPT2Config
from gpt2_engine import GPT2
from gpt_utils import (DataLoaderLite, DataProcessing, 
                       load_text, evaluate, get_batch,
                       get_lr)

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math

device = 'cpu'
if torch.cuda.is_available():
  device = 'cuda'
print(f"using device: {device}")

## model training
torch.manual_seed(1337)
if torch.cuda.is_available():
  torch.cuda.manual_seed(1337)

def train_gpt(config:GPT2Config, data, device):
  data_loader = DataProcessing(data)
  train_data, val_data, vocab_size = data_loader.encode_and_split(data, 0.8)

  total_batch_size = config.TOT_BATCH_SIZE
  B,T = config.BATCH_SIZE, config.SEQ_LEN

  assert total_batch_size %(B*T) == 0, "make sure total batch size is divisible by B*T"

  grad_accum_step = total_batch_size//(B*T)
  print(f"total desired batch size: {total_batch_size}")
  print(f"==> Calculated gradient Accum step: {grad_accum_step}")

  torch.set_float32_matmul_precision('high') #optim3 (for gpu to use tf32 for matmul ops) 
  model = GPT2(config, vocab_size=128).to(device)
  model = torch.compile(model) #optim1 (torch compiler)

  scaler = torch.amp.GradScaler('cuda')
  ## training loop
  #optimizer = torch.optim.AdamW(model.parameters(), lr = 3e-4, betas =[0.9,0.95], eps=1e-8)
  optimizer = model.configure_optimizers(weight_decay = 0.1, 
                                         learning_rate = 6e-4, 
                                         device = device)
  for step in range(config.MAX_STEPS):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0
    for _ in range(grad_accum_step):

      x,y = get_batch(train_data, config.BATCH_SIZE, config.SEQ_LEN)
      x,y = x.to(device), y.to(device)

    #  with torch.autocast(device_type=device, dtype=torch.float16): #optim2 (mixed precision)
      logits, loss = model(x,y)

      loss= loss/grad_accum_step
      loss_accum += loss.detach()

      scaler.scale(loss).backward()
      scaler.unscale_(optimizer)

    norm=torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) #gradient clipping
    lr = get_lr(step, config)
    for param_group in optimizer.param_groups:
      param_group['lr']= lr
    
    #optimizer.step()
    scaler.step(optimizer)
    scaler.update()
    #torch.cuda.synchronize()

    t1 = time.time()
    train_perp = math.exp(loss_accum)
    dt = (t1-t0)*1000 #ms
    toks_per_sec = (config.BATCH_SIZE * config.SEQ_LEN)/(t1-t0)

    if (step+1)%config.TRAIN_STEP==0:
      print(f"step: {step+1} | train_loss: {loss_accum:.2f} | train_perp: {train_perp:.2f} | norm: {norm:.4f} | lr: {lr:.4f} | dt: {dt:.2f}ms | toks/sec: {toks_per_sec:.0f}")

    if (step+1)%config.EVAL_STEP==0:
      val_loss = evaluate(model, val_data, 
                          config.BATCH_SIZE, 
                          config.SEQ_LEN, 
                          device)
      val_perp = math.exp(val_loss)

      print(f"step: {step+1} | val_loss: {val_loss:.2f} | val_perp: {val_perp:.2f}")

   # import sys; sys.exit()

    if (step+1)%config.GEN_EVAL ==0:
      #model = GPT2.pretrained('gpt2')
      model.eval()
      torch.manual_seed(42)
      if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
      
      #the prefill phase
      num_return_seq = 5
      text= "Hello, I'm a Language model,"
      tokens,_ = data_loader.encode_and_split(text, n=0)
      tokens = tokens.unsqueeze(0).repeat(num_return_seq, 1)

      x = tokens.to(device)
      print(x.shape)

      ##the prediction phase
      x = model.generate(x, config.MAX_NEW_TOKENS)

      for i in range(num_return_seq):
        tokens = x[i, :config.MAX_NEW_TOKENS].tolist()
        dec = data_loader.decode(tokens)
        print(">>>", dec)


def main():
  data = load_text("GPT2/wizard_of_oz.txt")
  config= GPT2Config()
  train_gpt(config, data, device)

if __name__ == "__main__":
  main()