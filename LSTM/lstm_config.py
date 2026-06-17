from dataclasses import dataclass
import torch

@dataclass
class LSTMConfig():
    N_HIDDEN:int = 128     #hidden size
    VOCAB_SIZE:int = 50257
    BATCH_SIZE =8
    SEQ_LEN = 64
    DIM = 128
    N_LAYER = 8
    DROPOUT=0.2
    WARMUP_STEPS= 120
    MAX_STEPS=3000
    MAX_NEW_TOKENS=100
    TRAIN_STEP=10
    EVAL_STEP=100
    GEN_EVAL=500
    TOT_BATCH_SIZE= 1024
    MAX_LR = 1e-2
    MIN_LR = 0.1 * MAX_LR
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
