# LSTM

A standard LSTM language model implementation.

## Overview

This project implements a LSTM-based language model from scratch using PyTorch:

- Standard LSTM cells with input, forget, and output gates
- Multi-layer LSTM stack
- Gradient clipping for training stability

## Model Architecture

```
Layers: 8 × 128 hidden dimension
```

**Key features:**
- **Multi-layer LSTM**: 8 stacked LSTM layers
- **Hidden State**: 128-dimensional hidden state
- **Dropout**: 0.2 dropout between layers

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Hidden dimension | 128 |
| Vocabulary size | 50257 |
| Layers | 8 |
| Dropout | 0.2 |
| Max sequence length | 64 |

## Requirements

```
torch
numpy
```

## Usage

### Training

Run training from the `LSTM` directory:

```bash
cd LSTM
python lstm_training_run.py
```

### Inference

The model includes a `generate()` method for text generation:

```python
model.eval()
generated_ids = model.generate(idx, max_new_tokens=100, temperature=0.8)
```

## Project Structure

```
LSTM/
├── lstm_engine.py       # Main LSTM model
├── lstm_architecture.py  # LSTM cell implementations
├── lstm_training_run.py  # Training loop
├── lstm_config.py     # LSTMConfig
└── lstm_utils.py    # Data loading utilities
```

## Acknowledgments

This implementation is based on standard LSTM architecture from:

- Hochreiter & Schmidhuber, "Long Short-Term Memory" (1997)
- Various LSTM language model implementations