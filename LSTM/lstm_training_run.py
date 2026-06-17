import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import torch.optim as optim
from torch.utils.data import DataLoader
import math

from lstm_engine import LSTM
from lstm_utils import estimate_loss, ChunkDataset, TextDataset, load_data, get_batch, evaluate
from lstm_config import LSTMConfig

## Training the Model
def training_run(data_dir, config, device,):
    torch.manual_seed(1443)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1024)

    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')

    #loading the dataset
    data = load_data(data_dir)
    data_loader = TextDataset(data)
    train_data, val_data, vocab_size = data_loader.encode_and_split_data(data, 0.8)

    ## loading data to dataloader
    train_ds = DataLoader(ChunkDataset(train_data, config.SEQ_LEN), 
                          batch_size=config.BATCH_SIZE, 
                          shuffle=True, num_workers=4,
                          pin_memory=True, prefetch_factor=2,
                          persistent_workers=True)
    
    val_ds = DataLoader(ChunkDataset(val_data, config.SEQ_LEN), 
                        batch_size=config.BATCH_SIZE, shuffle=False, 
                        num_workers=2, pin_memory=True, prefetch_factor=2,
                        persistent_workers=True)


    model = LSTM(n_embd= config.DIM, vocab_size = config.VOCAB_SIZE,
                n_hidden= config.N_HIDDEN, n_layers= config.N_LAYER,
                proj_size=config.VOCAB_SIZE, dropout =config.DROPOUT, 
                bias= True)

    model = model.to(device)
    model = torch.compile(model)
    parameter_size = sum(p.nelement() for p in model.parameters())
    print(f"==> Parameter Size: {parameter_size}")

    for p in model.parameters():
        p.requires_grad = True

    total_lossi, val_loss, total_norm, learning_rate = [],[], [], []

    optimizers = optim.AdamW(model.parameters(), lr= 1e-2)
    scheduler_cos = optim.lr_scheduler.CosineAnnealingLR(optimizers,  
                                                         T_max = config.MAX_STEPS, 
                                                         eta_min = config.MIN_LR) ## step decay

    # Initialize best validation loss for saving the model
    running_loss = 0.0
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported else torch.float16
    scaler = torch.amp.GradScaler('device')

    assert config.TOT_BATCH_SIZE % (config.BATCH_SIZE * config.SEQ_LEN) ==0, "Non divisible"
    grad_accum_step = config.TOT_BATCH_SIZE//(config.BATCH_SIZE * config.SEQ_LEN)
    print(f"==>Total_batches: {config.TOT_BATCH_SIZE}, mini_batches: {config.BATCH_SIZE}, seq_len: {config.SEQ_LEN}, Grad_accum_step: {grad_accum_step}")
    
    #setting up the training epoch
    for idx in range(config.MAX_STEPS):
        model.train()

        t0 = time.time()
        Xtr, Ytr = get_batch(train_data, config.BATCH_SIZE, config.SEQ_LEN)
        Xtr, Ytr = Xtr.to(device), Ytr.to(device)
        optimizers.zero_grad(set_to_none=True)

        if dtype == torch.bfloat16:
        
            for _ in range(grad_accum_step):
                with torch.autocast(device_type=device, dtype = dtype):
                    _, _, loss = model(Xtr, y = Ytr)
                    loss = loss/grad_accum_step
                    running_loss+=loss.detach()

                loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            total_norm.append(norm)
            optimizers.step()
            scheduler_cos.step()

        else:
            for _ in range(grad_accum_step):
                with torch.autocast(device_type=device, dtype=dtype):
                    _, _, loss = model(Xtr, y=Ytr)
                    loss = loss/grad_accum_step
                    running_loss+=loss.detach()
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizers)

            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            total_norm.append(norm)
            scaler.step(optimizers)
            scaler.update()
            scheduler_cos.step()

        current_lr = optimizers.param_groups[0]['lr']
        learning_rate.append(current_lr)
        #running_loss += loss.item()

        # The training process
        if (idx+1) % config.TRAIN_STEP == 0:
            avg_loss = running_loss / config.TRAIN_STEP
            tr_perp = math.exp(avg_loss)
            B, T = Xtr.shape
            dt = time.time() - t0
            toks_per_sec = (B * T) / dt

            # Update the progress bar display
            print(f"step: {idx+1} | tr_loss: {avg_loss:.4f} | tr_perp: {tr_perp:.2f} | norm: {norm:.2f} | lr: {current_lr:.2e} | dt : {dt*1000:.2f}ms | toks/sec : {toks_per_sec:.0f}tps")
            running_loss = 0.0  # Reset

            ## evaluating model
            # Detailed Evaluation every 1000 batches
        if (idx+1) % config.EVAL_STEP == 0:
            val_loss = evaluate(model, val_data, config.BATCH_SIZE, 
                                config.SEQ_LEN, device) # Update the epoch-end val loss
            val_perp = math.exp(val_loss)
            print(f"Epoch {idx+1} | Val Loss: {val_loss:.2f} | val_perp: {val_perp:.2f}")

        if (idx+1)% config.GEN_EVAL == 0:
            model.eval()
            num_return_seq=5
            text = "You are thou, "
            token, _ = data_loader.encode_and_split_data(text)
            token = token.unsqueeze(0).repeat(num_return_seq,1)

            x = token.to(device)
            print(x.shape)

            gen = model.generate(x, config.MAX_NEW_TOKENS)
            for i in range(num_return_seq):
                tokens = gen[i, :config.MAX_NEW_TOKENS].tolist()
                decoded_message = data_loader.decode(tokens)
                print(f"==> {decoded_message}")

            model.train()




def main():
    data_dir = "data/wizard_of_oz.txt"
    config = LSTMConfig(VOCAB_SIZE=128)
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    training_run(data_dir, config, device)

if __name__ =="__main__":
    main()