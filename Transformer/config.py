from dataclasses import dataclass
import torch

## assigning the parameters to variables
@dataclass
class TransformerConfig():
    batch_size:int = 8      # number of independent sequence to process in parallel
    max_iter:int = 1      #100k #total number of epoches
    block_size: int = 512     #the maximum context length of the prediction
    train_logging_interval: int = 500
    eval_interval: int =1000
    eval_iter:int = 100         
    n_embd: int= 512     #512     # Dimensions of the model (dmodel)
    n_head:int = 8     #8       # number of head in the multihead attention mechanism 
    n_layer:int= 6
    dropout: float= 0.1
    head_size: int = n_embd//n_head    #dimension of key and value dk==dv == dmodel/n_head
    learning_rate: float = 1e-1
    eps: float = 1e-9
    warmup_steps: int = 4000
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt_dir:str = 
    max_grad_norm: float = 1.0
    max_tokens_per_batch:int = batch_size * block_size