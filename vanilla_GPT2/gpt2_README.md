# Vanilla GPT-2

A GPT-2 language model implementation based on "Language Models are Unsupervised Multitask Learners" (Radford et al., 2019).

## Overview

This project implements a decoder-only Transformer (GPT) language model from scratch using PyTorch. It includes:

- Causal (unidirectional) attention
- Weight tying between embedding and output projection
- Gradient accumulation for effective larger batch sizes
- Mixed precision training support

## Model Architecture

The GPT-2 follows the original GPT paper:

```
Layers: 8 × 128 dim × 8 heads
```

**Key features:**
- **Causal Attention**: Each token can only attend to previous tokens
- **Weight Tying**: Input embedding weights shared with output projection
- **Efficient Attention**: Uses `F.scaled_dot_product_attention`
- **Pre-LayerNorm**: Applied in each block for better gradient flow

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Embedding dimension | 128 |
| Attention heads | 8 |
| Layers | 8 |
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

Run training from the `vanilla_GPT2` directory:

```bash
cd vanilla_GPT2
python training_run.py
```

Training uses mixed precision and torch.compile() for optimization.

### Inference

The model includes a `generate()` method for text generation:

```python
model.eval()
generated_ids = model.generate(idx, max_new_tokens=100, temperature=0.8, top_k=40)
decoded = tokenizer.decode(generated_ids[0])
```

## Project Structure

```
vanilla_GPT2/
├── gpt2_engine.py       # Main GPT2 model
├── model_architecture.py    # TransformerBlock
├── training_run.py     # Training loop
├── gpt2_config.py      # GPT2Config
├── gpt_utils.py        # Data loading utilities
└── profiler.py        # Performance profiling
```

## Learning Rate Schedule

Uses a linear warmup schedule:

```
lr = min_lr + (max_lr - min_lr) × step / warmup_steps  (during warmup)
lr = max_lr × 0.1^ (step / max_steps)              (after warmup)
```

With `warmup_steps = 120`, `max_lr = 6e-4`.

## Acknowledgments

This implementation is based on:

- Radford et al., "Language Models are Unsupervised Multitask Learners" (2019)
- https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf