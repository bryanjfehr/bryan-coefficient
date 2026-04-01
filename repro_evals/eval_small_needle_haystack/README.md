# Small Model Needle-in-a-Haystack Evaluation

This folder contains the necessary scripts to reproduce the λB (Bryan Coefficient) discontinuity mapping for small language models (SLMs) such as Qwen2-1.5B and Phi-3-mini.

## Hardware Tiers
- **Local (RTX 4060)**: Baseline testing for memory efficiency and 4-bit quantization.
- **Google Colab (L4 GPU)**: Standard evaluation tier for mapping the discontinuity threshold across 8k-12k context windows.

## Methodology
The evaluation uses a "Needle-in-a-Haystack" (Passkey Retrieval) benchmark combined with WikiText-103 Perplexity (PPL) measurement. We surgically alter the attention mechanism of the target models to apply a distance-decay penalty ($\lambda_B$) and physically evict low-importance KV-cache tokens.

### Key Parameters
- `sink_protection`: 20 tokens (ensures foundational attention anchors are preserved).
- `local_window`: 128 tokens (ensures recent context is retained).
- `eviction_threshold`: 1e-3 (minimum attention score sum required for token survival).

## Reproduction Steps in Google Colab
1. Open a new Google Colab notebook with an **L4 GPU** runtime.
2. Run the provided `repro_small_needle.py` script.
3. The script will:
   - Install dependencies (transformers, bitsandbytes, etc.).
   - Load models directly from HuggingFace.
   - Apply the Bryan Coefficient surgery.
   - Sweep through λB values from 0.0 to 0.01.
   - Output accuracy and VRAM metrics.

## Files
- `repro_small_needle.py`: Self-contained reproduction script.
- `metadata.json`: Technical parameters for this evaluation tier.
