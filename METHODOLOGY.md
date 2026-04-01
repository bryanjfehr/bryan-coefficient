# Methodology: The Bryan Coefficient ($\lambda_B$) and Context Compression

This document provides a rigorous mathematical and procedural foundation for the research conducted in the Bryan Coefficient evaluation suite. It details the mechanisms used to achieve massive KV-cache compression in Small Language Models (SLMs) without triggering logical collapse.

## 1. Mathematical Foundation: The Bryan Coefficient ($\lambda_B$)

The Bryan Coefficient ($\lambda_B$) is defined as a continuous, distance-based penalty applied to the attention scores before the softmax normalization. It acts as a high-pass filter for attention, effectively denoising the query path by penalizing older, potentially irrelevant context.

### 1.1 Attention Score Modification

In a standard Transformer attention layer, the attention scores $A$ for a query at position $i$ and a key at position $j$ are calculated as:

$$A_{i,j} = \frac{Q_i K_j^T}{\sqrt{d_k}}$$

The Bryan Coefficient modifies this by injecting a temporal decay penalty $P_{i,j}$:

$$\tilde{A}_{i,j} = A_{i,j} + P_{i,j}$$

Where $P_{i,j}$ is defined as:

$$P_{i,j} = - \lambda_B \cdot \Delta_{i,j} \cdot M_{i,j}$$

- $\lambda_B$: The Bryan Coefficient ($0 \le \lambda_B$).
- $\Delta_{i,j}$: The causal distance between the query and the key, $\Delta_{i,j} = i - j$.
- $M_{i,j}$: A protection mask that ensures foundational tokens are not penalized.

### 1.2 Attention Sink Protection

To prevent logical collapse, "Attention Sinks" (foundational tokens like the start-of-sequence token) are protected from the $\lambda_B$ penalty and subsequent eviction. The mask $M_{i,j}$ is defined as:

$$M_{i,j} = 
\begin{cases} 
0 & \text{if } j < S_{len} \\
0 & \text{if } i - j < W_{local} \\
1 & \text{otherwise}
\end{cases}$$

- $S_{len}$: The number of protected "sink" tokens (default: 20-2048).
- $W_{local}$: The size of the local attention window (default: 128).

## 2. Physical KV-Cache Eviction

The modified attention weights $\tilde{A}_{i,j}$ are normalized via softmax to produce attention probabilities $\alpha_{i,j}$:

$$\alpha_{i,j} = \text{softmax}(\tilde{A}_{i,j})$$

The importance of a token $j$ is calculated as the sum of its attention probabilities across all queries $i$ and attention heads $h$:

$$I_j = \sum_{h} \sum_{i} \alpha_{i,j,h}$$

A token $j$ is physically evicted from the KV-cache if its importance falls below the eviction threshold $\epsilon$ and it is not protected by the sink mask:

$$\text{Evict } j \text{ if } I_j < \epsilon \text{ AND } j \text{ is not protected}$$

## 3. Discontinuity Mapping: Finding the "Amnesia Wall"

The "Amnesia Wall" is the exact mathematical threshold ($\lambda_{B, cliff}$) where the model's performance (measured by Passkey Retrieval accuracy) undergoes a sharp phase transition from 100% to 0%.

### 3.1 High-Resolution Bisection Search

To isolate this cliff with high precision ($\pm 0.00001$), we employ a micro-sweep bisection search algorithm:

1. Define a search range $[\lambda_{B, low}, \lambda_{B, high}]$.
2. Test the model at the midpoint $\lambda_{B, mid}$.
3. If accuracy is 100%, set $\lambda_{B, low} = \lambda_{B, mid}$.
4. If accuracy is 0%, set $\lambda_{B, high} = \lambda_{B, mid}$.
5. Repeat until the resolution limit is reached.

### 3.2 Curvature Analysis

We analyze the second derivative of the performance-to-compression curve to identify the "elbow" where VRAM recovery is maximized just before foundational attention sinks are destroyed.

## 4. Academic Citations

This research builds upon several foundational techniques in Transformer optimization:

- **Attention Sinks**: [StreamingLLM] Xiao et al. (2023). "Efficient Streaming Language Models with Attention Sinks."
- **KV-Cache Eviction**: [H2O] Zhang et al. (2023). "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models."
- **Quantization**: [QLoRA] Dettmers et al. (2023). "QLoRA: Efficient Finetuning of Quantized LLMs."
- **FlashAttention**: Dao et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness."
- **Lost in the Middle**: Liu et al. (2023). "Lost in the Middle: How Language Models Use Long Contexts."

## 5. Experimental Setup & Hardware Tiers

The evaluation gauntlet is tiered across three hardware environments to balance local accessibility with high-compute requirements for large models.

### 5.1 Tier 1: Local Small Model Baseline
- **Hardware**: NVIDIA RTX 4060 (8GB VRAM).
- **Target Models**: Qwen2-1.5B, Phi-3-mini.
- **Constraints**: Strict 8GB VRAM ceiling mandates the use of 4-bit quantization (NF4) and `attn_implementation="eager"`.
- **Purpose**: Rapid iteration of $\lambda_B$ gradients and initial discontinuity mapping.

### 5.2 Tier 2: Standard Agentic Evaluation (v1-v4)
- **Hardware**: Google Colab (NVIDIA L4 GPU - 24GB VRAM).
- **Target Models**: Qwen2.5-1.5B, Phi-3, Gemma-2-2B, Llama-3.2-3B.
- **Framework**: LangGraph SQL State Machine for multi-turn reasoning evaluation.
- **Purpose**: Measuring the impact of KV-cache pruning on long-context agentic reasoning and tool-use stability.

### 5.3 Tier 3: Large Model High-Resolution Gauntlet
- **Hardware**: Google Colab (NVIDIA A100 GPU - 40GB/80GB VRAM).
- **Target Models**: GPT-OSS-20B, Qwen3-30B, Llama-3.1-8B (High Context).
- **Strategy**: High-resolution bisection search ($\pm 0.00001$ precision) to find the absolute physical limit of context compression.
- **Purpose**: Proving architecture viability for massive context windows in production-grade SLMs.
