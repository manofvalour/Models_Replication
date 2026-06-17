# Models_Replication

Replicating research papers and building models from scratch.

## Overview

This repository contains implementations of foundational deep learning models from research papers. The goal is to understand core architectures by implementing them from scratch using PyTorch.

## Projects

### Transformer (English-French Translation)

A vanilla Transformer implementation for sequence-to-sequence translation based on "Attention Is All You Need" (Vaswani et al., 2017).

**Architecture:**
- Encoder-Decoder Transformer with 6 layers
- 512 embedding dimension, 8 attention heads
- Pre-LayerNorm, weight tying, label smoothing

**Run:**
```bash
cd Transformer && python model_training.py
```

### Vanilla GPT-2

A GPT-2 language model implementation based on "Language Models are Unsupervised Multitask Learners" (Radford et al., 2019).

**Architecture:**
- Decoder-only Transformer with 8 layers
- 128 embedding dimension, 8 attention heads
- Causal (unidirectional) attention

**Run:**
```bash
cd vanilla_GPT2 && python training_run.py
```

### MoE with KV Cache

A GPT with Mixture of Experts and KV cache optimization for efficient inference.

**Architecture:**
- GPT architecture with MoE layers (4 experts, top-2 routing)
- KV cache during generation for O(1) token generation
- Load balancing auxiliary loss

**Run:**
```bash
cd MOE_with_KV_cache && python training_run.py
```

### LSTM

A standard LSTM language model.

**Run:**
```bash
cd LSTM && python lstm_training_run.py
```

## Installation

```bash
pip install torch transformers datasets sacrebleu wandb tqdm numpy
```

## Papers Implemented

- **Transformer**: [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (2017)
- **GPT-2**: [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf) (2019)