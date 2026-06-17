# MoE with KV Cache

A GPT model with Mixture of Experts (MoE) and KV cache optimization for efficient inference.

## Overview

This project implements a GPT architecture with Mixture of Experts layers and KV cache optimization:

- **Mixture of Experts**: 4 experts with top-2 routing
- **KV Cache**: Caches key-value pairs during generation for O(1) per-token latency
- **Load Balancing**: Auxiliary loss to ensure expert utilization
- **Efficient Inference**: Uses cached KV for autoregressive generation

## Model Architecture

```
Layers: 8 × 128 dim × 8 heads
Experts: 4 (top-2 routing)
```

**Key features:**
- **MoE Layer**: Each token routed to top-2 of 4 experts
- **KV Cache**: Stores cached keys/values during generation
- **Load Balancing Loss**: Ensures even expert utilization
- **Weight Tying**: Input embedding weights shared with output projection

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Embedding dimension | 128 |
| Attention heads | 8 |
| Layers | 8 |
| Number of experts | 4 |
| Top-k routing | 2 |
| Feed-forward dimension | 512 (4 × dim) |
| Dropout | 0.2 |
| Max sequence length | 64 |

## Requirements

```
torch
numpy
```

## Usage

### Training

Run training from the `MOE_with_KV_cache` directory:

```bash
cd MOE_with_KV_cache
python training_run.py
```

Training uses MoE layers with auxiliary load balancing loss.

### Inference with KV Cache

The model includes a `generate()` method with KV cache support:

```python
model.eval()
generated_ids = model.generate(
    idx,
    max_new_tokens=100,
    temperature=0.8,
    top_k=40,
    use_cache=True  # Enable KV cache
)
```

KV cache is automatically managed - during generation, only the new token is processed while attending to cached keys/values from previous tokens.

## Project Structure

```
MOE_with_KV_cache/
├── moe_engine.py       # Main MOEGPT model
├── moe_architecture.py   # CausalMultiheadAttn, MoELayer, TransformerBlock
├── training_run.py    # Training loop
├── moe_config.py     # MOEConfig
└── moe_utils.py     # Data loading utilities
```

## MoE Details

### Router
- Linear layer maps input to expert logits
- Softmax over expert probabilities
- Top-2 experts selected per token

### Expert
- Two-layer MLP with GELU activation
- Input projection: dim → 4×dim
- Output projection: 4×dim → dim

### Load Balancing Loss
```
load_loss = num_experts × (importance × load).sum()
```
Where importance is mean router probability and load is fraction of tokens to each expert.

## Learning Rate Schedule

Uses a linear warmup schedule:

```
lr = min_lr + (max_lr - min_lr) × step / warmup_steps  (during warmup)
lr = max_lr × 0.1^ (step / max_steps)              (after warmup)
```

With `warmup_steps = 120`, `max_lr = 6e-4`.