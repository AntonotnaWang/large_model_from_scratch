# large_model_from_scratch

**English** | [中文版 (Chinese)](README-zh.md)

Learning the key components of LLMs/VLMs by building them from scratch.

This repo contains a single, self-contained Jupyter notebook that builds and trains a **"small but complete"** large language model from the ground up — a tiny character-level GPT trained on Shakespeare — while walking through the key techniques behind modern LLM training and inference, from tokenization all the way to KV-cache decoding.

## Notebook

| Language | Notebook |
| :-- | :-- |
| English (default) | [`nano_llm_from_scratch-en.ipynb`](nano_llm_from_scratch-en.ipynb) |
| 中文 (Chinese) | [`nano_llm_from_scratch-zh.ipynb`](nano_llm_from_scratch-zh.ipynb) |

Each cell is meant to be read and run top to bottom; the prose explains the *why*, and the code is kept minimal so you can see the *how*.

## What you'll learn

The notebook is organized as a guided build-up:

- **0. Introduction to LLMs** — next-token prediction, and what "large" really means (scaling laws, emergence).
- **1. Data preparation** — download & inspect the tiny-Shakespeare corpus; **tokenization** two ways: character-level, and (optional) **byte-level BPE** with a from-the-ground-up explanation of how GPT's tokenizer actually works.
- **2. Hyperparameters** — the handful of knobs for our toy model.
- **3. Dataset loader** — building `(input, target)` pairs for next-token prediction.
- **4. Embedding** — semantic (token) embedding, plus three position encodings: **learned absolute**, **sinusoidal**, and **RoPE** (rotary).
- **5. Attention** — self-attention from first principles, then **multi-head attention**, **KV cache**, **grouped-query attention (GQA)**, and **multi-head latent attention (MLA)**; the **feed-forward layer** (ReLU → GELU → SwiGLU → **MoE**); and **Add & LayerNorm** (with **RMSNorm** and pre-/post-norm).
- **6. Building the model** — assembling the Transformer, plus the engineering tricks real training needs (dropout, weight tying, initialization, LR scheduling, gradient clipping, special tokens) and **why cross-entropy** is the loss.
- **7. Training** — the training loop, teacher forcing vs. autoregressive generation, warmup + cosine LR schedule, and validation/perplexity.
- **8. Inference** — **decoding strategies** (greedy, temperature, top-k, nucleus/top-p, beam search), the default **KV-cache generation loop**, EOS handling, and a map of what production LLM inference additionally requires (continuous batching, PagedAttention, speculative decoding, quantization, parallelism, …).

Modern-practice callouts throughout connect each toy component to what Llama 3 / Qwen / DeepSeek / Gemma / Mistral actually use.

## The model at a glance

A deliberately tiny configuration so everything trains in minutes on a single GPU (or even CPU):

| Setting | Value |
| :-- | :-- |
| Tokenizer | character-level (vocab size = 65) |
| Context length (`block_size`) | 16 |
| Embedding dim (`n_embed`) | 32 |
| Layers (`n_layers`) | 16 |
| Attention heads (`num_heads`) | 8 |
| Head size (`head_size`) | 16 |
| Batch size | 128 |
| Learning rate | 1e-4 |

## Setup

```bash
pip install torch tiktoken matplotlib tqdm numpy
```

- `torch` — the only hard dependency for the model itself.
- `tiktoken` — optional, only for the byte-level BPE section.
- `matplotlib`, `tqdm`, `numpy` — plotting, progress bars, and small utilities used along the way.

## How to run

```bash
jupyter notebook nano_llm_from_scratch-en.ipynb
```

Run the cells from top to bottom. On the first run, the tiny-Shakespeare dataset is downloaded automatically into [`data/`](data/). A GPU is used if available; otherwise it falls back to CPU.

## Repository layout

```
large_model_from_scratch/
├── README.md                       # this file (English)
├── README-zh.md                    # Chinese version
├── nano_llm_from_scratch-en.ipynb  # the tutorial (English)
├── nano_llm_from_scratch-zh.ipynb  # the tutorial (Chinese)
└── data/                           # dataset (auto-downloaded on first run)
```

## Acknowledgements

The character-level setup and the tiny-Shakespeare dataset are inspired by Andrej Karpathy's work ([char-rnn](https://github.com/karpathy/char-rnn), [nanoGPT](https://github.com/karpathy/nanoGPT)). Byte-level BPE uses OpenAI's [`tiktoken`](https://github.com/openai/tiktoken).
