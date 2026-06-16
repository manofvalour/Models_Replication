# Models_Replication

A vanilla Transformer implementation for English-French sequence-to-sequence translation, replicating the "Attention Is All You Need" architecture from the seminal paper.

## Overview

This project implements a complete encoder-decoder Transformer model from scratch using PyTorch. It includes:

- Distributed training across multiple GPUs
- Mixed precision training (bfloat16/float16)
- Gradient accumulation for effective larger batch sizes
- BLEU score evaluation
- Checkpointing with Weights & Biases integration

## Model Architecture

The Transformer follows the original "Attention Is All You Need" paper:

```
Encoder: 6 layers × 512 dim × 8 heads
Decoder: 6 layers × 512 dim × 8 heads
```

**Key features:**
- **Pre-LayerNorm**: Applied in each block for better gradient flow and training stability
- **Weight Tying**: Input embedding weights are shared with the output projection layer
- **Efficient Attention**: Uses `F.scaled_dot_product_attention` for memory-efficient computation
- **Label Smoothing**: 0.1 smoothing in cross-entropy loss

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Embedding dimension (d_model) | 512 |
| Attention heads | 8 |
| Layers per stack | 6 |
| Feed-forward dimension | 2048 (4 × d_model) |
| Dropout | 0.1 |
| Max sequence length | 512 |

## Requirements

```
torch
transformers
datasets
sacrebleu
wandb
tqdm
numpy
```

Install dependencies:

```bash
pip install torch transformers datasets sacrebleu wandb tqdm numpy
```

## Usage

### Training

Run training from the `Transformer` directory:

```bash
cd Transformer
python model_training.py
```

Training automatically uses all available GPUs via PyTorch DistributedDataParallel (DDP). The model will:

1. Load the English-French translation dataset from HuggingFace
2. Initialize a T5-base tokenizer
3. Train with the learning rate schedule from the original paper (warmup + inverse square root decay)
4. Log metrics to Weights & Biases
5. Save checkpoints to `checkpoints/`

### Resume Training

To resume from a checkpoint, modify `model_training.py` and pass the checkpoint path to the `main()` function:

```python
main(0, 1, model_location='checkpoints/best_model.pt', resume_training=True)
```

## Learning Rate Schedule

The model uses the original "Attention Is All You Need" learning rate schedule:

```
lr = d_model^(-0.5) × min(step^(-0.5), step × warmup^(-1.5))
```

With `warmup_steps = 4000`.

## Data

- **Dataset**: `sjsurbhi/english-to-french-translation` from HuggingFace
- **Tokenizer**: T5-base
- **Filtering**: Sequences longer than `block_size` (512) tokens are filtered out
- **Batching**: Custom `DistributedWMTBatchSampler` creates variable-length batches bounded by `max_tokens_per_batch`

## Checkpoints

Checkpoints are saved to:
- `checkpoints/best_model.pt` - Best model by validation loss
- `checkpoints/latest.pt` - Most recent checkpoint for recovery

Each checkpoint contains:
- Model state dict
- Optimizer state dict
- Scheduler state dict
- Training/validation loss
- Weights & Biases run ID

## Inference

The model includes a `generate()` method for inference:

```python
model.eval()
generated_ids = model.generate(src, max_len=150, sos_id=1, eos_id=2)
decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
```

## Project Structure

```
Models_Replication/
├── Transformer/
│   ├── transformer_engine.py  # Main Transformer model
│   ├── model_architecture.py   # MultiHeadAttention, Block
│   ├── model_training.py        # Training loop
│   ├── config.py            # TransformerConfig
│   ├── utils.py             # Data loading, BLEU, checkpoints
│   └── checkpoints/       # Saved model checkpoints
└── README.md
```

## Acknowledgments

This implementation is based on:

- Vaswani et al., "Attention Is All You Need" (2017)
- https://arxiv.org/abs/1706.03762