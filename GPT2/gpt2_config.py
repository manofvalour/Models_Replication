from dataclasses import dataclass

@dataclass
class GPT2Config():
    vocab_size:int = 50257
    batch_size: int=5
    #drop_out:float
    n_layers:int = 12
    n_embd: int = 768
    #max_lr:float = 2.5e-4
    block_size:int = 512
    n_head: int = 12
    #max_epoch:int = 100