from tkinter import FALSE
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import math

from moe_utils import DataProcessing, load_text, evaluate, get_batch
from moe_engine import MOEGPT
from moe_config import MOEConfig
device = 'cuda' if torch.cuda.is_available() else 'cpu';




## Data and target
def train_loop(config, text_file):
    data = DataProcessing(text_file)
    train, val, _ = data.encode_and_split(text_file, 0.8);
    
    model = MOEGPT(config, allow_moe=True).to(device)
    model = torch.compile(model)
    print(f"model size: {model.get_num_params()}")
    torch.set_float32_matmul_precision("high")

    optimizer = optim.AdamW(model.parameters(), lr=config.MAX_LR, weight_decay=0.1, fused=True)
    scaler = torch.amp.GradScaler('cuda')
    #num_batches = len(train)//(BATCH_SIZE*SEQ_LEN)
    #print(num_batches)
    num_batches = 200
    total_batch_size = config.TOT_BATCH_SIZE
    assert total_batch_size% (config.BATCH_SIZE*config.SEQ_LEN) ==0
    grad_accum_step = total_batch_size//(config.BATCH_SIZE*config.SEQ_LEN)

    print(f"==>Total Batch Size: {total_batch_size}, Grad accum step: {grad_accum_step}")

    for epoch in range(config.MAX_STEPS):
        total_loss = 0.0
        aux_total_loss = 0.0
        
        model.train()
        start = time.time()
        optimizer.zero_grad(set_to_none =True)
        total_loss=0.0
        aux_total_loss=0.0
       
        for _ in range(grad_accum_step):
            X,y = get_batch(train, config.BATCH_SIZE, config.SEQ_LEN)
            # X = X.pin_memory().to(device,non_blocking=True)
            #y = y.pin_memory().to(device, non_blocking=True)

            with torch.amp.autocast('cuda', dtype=torch.float16):
                logit, loss, aux_loss = model(X,y)
            
            total = loss/grad_accum_step + aux_loss/grad_accum_step
            total_loss +=total.item()
            aux_total_loss+=aux_loss.item()

            scaler.scale(loss).backward()
        
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0)

        scaler.step(optimizer)
        scaler.update()

        if (epoch+1)%config.TRAIN_STEP ==0:
            avg_loss = total_loss
            avg_aux_loss = aux_total_loss
            train_perp= math.exp(avg_loss)
            end = time.time()
            time_change = end-start
            toks_per_sec = (config.BATCH_SIZE * config.SEQ_LEN)/time_change

            print(f"epoch: {epoch+1}/{config.MAX_STEPS} | train_loss: {avg_loss:.2f} | train_perp: {train_perp:.2f} | aux_loss: {avg_aux_loss:.2f} | dt: {time_change:.4f}sec | toks/sec: {toks_per_sec:.0f} ")

        if (epoch+1)%config.EVAL_STEP ==0:
            val_loss = evaluate(model,config,val,device)
            val_perp = math.exp(val_loss)
            print(f"epoch: {epoch+1}/{config.MAX_STEPS} | val_loss: {val_loss:.2f} | val_perp: {val_perp:.2f}")

        if (epoch+1)%config.GEN_STEP ==0:
            text = "Having  this  thought  in  mind"
            text_enc,  _ = data.encode_and_split(text, 0.0)
            idx = text_enc.detach().clone().to(dtype=torch.long, device=device).unsqueeze(0)
            start= time.time()
         #   model.reset_cache()
            da = model.generate(idx, max_new_tokens=200, top_k =40)
            gen_text = data.decode(da[0].tolist())
            print(gen_text)
            end= time.time()
            print(f'time_taken: {(end - start):.2f}secs')
            
            start1= time.time()
          #  model.reset_cache()
            dam = model.generate(idx, max_new_tokens=200, top_k =40, use_cache=False)
            gen_text2 = data.decode(dam[0].tolist())
            print(gen_text2)
            end1 = time.time()
            print(f'time_taken_no_cache: {(end1 - start1):.2f}secs')

if __name__ == "__main__":
    text_file = load_text("data/wizard_of_oz.txt");
    config= MOEConfig(VOCAB_SIZE = 128)
    train_loop(config,text_file)
