# λB Eager Mode Surgery: Technical Documentation

## 1. Executive Summary
This track successfully implemented the "Intercept" protocol for Small Language Models (SLMs) using PyTorch monkey-patching. By surgically altering the `forward` pass of attention layers, we enabled continuous temporal decay and post-softmax KV-cache pruning, fulfilling the core requirements for mapping the Bryan Coefficient ($\lambda_B$) Discontinuity Threshold.

## 2. Infrastructure: The Monkey-Patch
We established a robust monkey-patching utility in `surgery_utils.py`. This allows for runtime replacement of model methods without altering the underlying library source code.
- **`patch_attention`**: Binds a custom `bryan_coefficient_forward` to specific model layers.
- **`unpatch_attention`**: Restores the original forward pass, ensuring experimental isolation.

## 3. Surgical Interventions

### 3.1 Temporal Decay Penalty
Implemented `apply_decay_penalty` to modify `attn_weights` immediately prior to softmax.
- **Mechanism**: Calculates a temporal distance matrix $\Delta t = i - j$ between queries and keys.
- **Logic**: Applies a subtractive penalty scaled by $\lambda_B$, favoring recent context and preparing "dead weights" for eviction.

### 3.2 KV-Cache Pruning (Garbage Collection)
Implemented `prune_kv_cache` to physically reduce the VRAM footprint of the KV-cache.
- **Scoring**: Uses post-softmax attention scores to identify low-value vertices.
- **Attention Sink Protection**: A configurable `sink_count` (default: 4) ensures foundational tokens are never evicted, preventing logical collapse.
- **Surgery**: Physically slices the Key and Value tensors and updates the model's `Cache` object in-place.

## 4. Integration & Targeting
The surgery was integrated into `run_benchmarks.py` via the `apply_bryan_coefficient` function.
- **Selective Patching**: The implementation specifically targets top-level attention modules (e.g., `self_attn`) while ignoring internal projection layers.
- **Architecture Support**: Verified compatibility with both Llama-3 and Qwen2 architectures.

## 5. Verification Results
- **Baseline Stability ($\lambda_B = 0.00$)**: Verified successful passkey retrieval (100% accuracy) with all attention layers patched, confirming no functional regressions in the default state.
- **VRAM Isolation**: Confirmed that the prefill and generation phases operate within the 8GB memory ceiling for context windows up to 8,000 characters.

## 6. Usage
The surgery is automatically triggered during benchmark execution:
```bash
./venv/bin/python3 run_benchmarks.py --model <model_id> --lambda_b <value>
```
The console output will log the number of patched layers and the status of the surgery protocol.
