from dataclasses import dataclass
import torch

@dataclass
class MOEConfig():
    VOCAB_SIZE:int = 50257
    MAX_LR:float = 6e-4
    MIN_LR:float =  MAX_LR *0.1
    BATCH_SIZE =8
    SEQ_LEN = 64
    DIM = 128
    N_HEAD = 8
    N_LAYER = 8
    DROPOUT=0.2
    HEAD_SIZE= DIM//N_HEAD
    WARMUP_STEPS= 120
    MAX_STEPS=3000
    MAX_NEW_TOKENS=100
    TRAIN_STEP=10
    EVAL_STEP=100
    GEN_STEP=50
    TOT_BATCH_SIZE= 1024
    NUM_EXPERT = 4
    TOP_K =2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
