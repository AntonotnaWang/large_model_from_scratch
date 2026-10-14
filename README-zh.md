# large_model_from_scratch

[English (英文版)](README.md) | **中文**

通过从零开始亲手构建，来学习 LLM/VLM 的关键组件。

本仓库包含一个自包含的 Jupyter notebook，从零开始构建并训练一个**“麻雀虽小、五脏俱全”**的大语言模型——一个在 Shakespeare 文本上训练的字符级小型 GPT——并沿途讲清现代 LLM 训练与推理背后的关键技术，从 tokenization 一直讲到 KV cache 解码。

## Notebook

| 语言 | Notebook |
| :-- | :-- |
| English(默认) | [`nano_llm_from_scratch-en.ipynb`](nano_llm_from_scratch-en.ipynb) |
| 中文 | [`nano_llm_from_scratch-zh.ipynb`](nano_llm_from_scratch-zh.ipynb) |

每个 cell 都建议自上而下地阅读并运行；文字讲清“为什么”，代码尽量精简，让你看清“怎么做”。

## 你将学到什么

整个 notebook 按照循序渐进的方式组织：

- **0. 大语言模型简介** —— next-token prediction，以及“大”到底意味着什么(scaling law、涌现)。
- **1. 数据准备** —— 下载并检查 tiny-Shakespeare 数据集；两种 **tokenization**：字符级，以及(可选的)**字节级 BPE**，并从头讲清 GPT 的分词器究竟是怎么工作的。
- **2. 超参数** —— 我们这个玩具模型的几个关键旋钮。
- **3. Dataset loader** —— 为 next-token prediction 构造 `(input, target)` 样本对。
- **4. Embedding** —— 语义(token)embedding，以及三种位置编码：**自学习绝对位置**、**Sinusoidal**、以及 **RoPE**(旋转位置编码)。
- **5. Attention** —— 从第一性原理讲 self-attention，再到 **multi-head attention**、**KV cache**、**grouped-query attention (GQA)** 与 **multi-head latent attention (MLA)**；**feed-forward layer**(ReLU → GELU → SwiGLU → **MoE**)；以及 **Add & LayerNorm**(含 **RMSNorm** 与 pre-/post-norm)。
- **6. 构建模型** —— 组装 Transformer，加上真实训练所需的工程技巧(dropout、权重绑定、初始化、学习率调度、梯度裁剪、特殊 token)，以及**为什么用交叉熵**作损失函数。
- **7. 训练** —— 训练循环、teacher forcing 与自回归生成的对比、warmup + 余弦退火学习率调度，以及 validation/perplexity。
- **8. 推理技巧** —— **解码策略**(greedy、temperature、top-k、nucleus/top-p、beam search)、默认启用的 **KV-cache 生成循环**、EOS 处理，以及一张“生产级 LLM 推理还需要什么”的技术地图(continuous batching、PagedAttention、推测解码、量化、并行……)。

全程穿插“现代实践”提示，把每个玩具组件和 Llama 3 / Qwen / DeepSeek / Gemma / Mistral 等真实模型的做法对应起来。

## 模型一览

一个刻意做得极小的配置，让一切都能在单卡(甚至 CPU)上几分钟内训练完成：

| 配置项 | 取值 |
| :-- | :-- |
| Tokenizer | 字符级(词表大小 = 65) |
| 上下文长度(`block_size`) | 16 |
| Embedding 维度(`n_embed`) | 32 |
| 层数(`n_layers`) | 16 |
| 注意力头数(`num_heads`) | 8 |
| 每个 head 的维度(`head_size`) | 16 |
| Batch size | 128 |
| 学习率 | 1e-4 |

## 环境准备

```bash
pip install torch tiktoken matplotlib tqdm numpy
```

- `torch` —— 模型本身唯一的硬性依赖。
- `tiktoken` —— 可选，仅用于字节级 BPE 一节。
- `matplotlib`、`tqdm`、`numpy` —— 画图、进度条，以及沿途用到的一些小工具。

## 如何运行

```bash
jupyter notebook nano_llm_from_scratch-zh.ipynb
```

自上而下运行各个 cell。首次运行时，tiny-Shakespeare 数据集会被自动下载到 [`data/`](data/) 目录。有 GPU 时会自动使用 GPU，否则回退到 CPU。

## 仓库结构

```
large_model_from_scratch/
├── README.md                       # 英文版(默认)
├── README-zh.md                    # 本文件(中文版)
├── nano_llm_from_scratch-en.ipynb  # 教程(英文)
├── nano_llm_from_scratch-zh.ipynb  # 教程(中文)
└── data/                           # 数据集(首次运行自动下载)
```

## 致谢

字符级的设置与 tiny-Shakespeare 数据集受 Andrej Karpathy 的工作启发([char-rnn](https://github.com/karpathy/char-rnn)、[nanoGPT](https://github.com/karpathy/nanoGPT))。字节级 BPE 使用 OpenAI 的 [`tiktoken`](https://github.com/openai/tiktoken)。
