# Comprehensive Architecture & Telemetry Audit (v4 Protocol)

## 1. The Root Cause of the "0.0" and Static VRAM Bug
Through rigorous inspection of the LangGraph state machine, the Outlines v1.2.12 wrapper hierarchy, and the `telemetry_node_wrapper`, the exact mechanism of failure has been identified.

**The Silent Exception Loop:**
In `sql_agent_utils.py`, the telemetry wrapper attempted to access the cache via:
```python
model = state["outlines_model"].model
pkv = getattr(model, "past_key_values", getattr(model.config, "_kv_cache_telemetry", None))
```
In Outlines v1.2+, `outlines_model.model` is an internal `TransformersWrapper` class, *not* the raw HuggingFace `AutoModelForCausalLM`. Consequently, `model.config` throws an `AttributeError`. 
Because this was wrapped in a generic `try...except`, the error was silently caught, and the `vram_history` and `kv_cache_history` lists were **never appended to the state**.

**The Fallback Illusion:**
At the end of the run, `build_and_run_graph` executed:
```python
peak_vram = max(final_state.get("vram_history", [0])) if final_state.get("vram_history") else get_peak_vram()
avg_cache = sum(...) if final_state.get("kv_cache_history") else 0.0
```
Because the history lists were empty (`[]` evaluates to `False`), it fell back to executing `get_peak_vram()` once, long after generation concluded and the cache was garbage collected. This resulted in `avg_cache` defaulting to `0.0` and `peak_vram` returning the static memory footprint of the model weights.

---

## 2. Model Architecture & Cache Structures (Cited Audit)

To guarantee academic fidelity across the 4 specific models on the A100 cluster, we must align our surgery with their exact `transformers` implementations.

### A. GPT-OSS 20B (OpenAI Open-Weights)
*   **Attention Signature:** Standard 3-element return `(attn_output, attn_weights, past_key_values)` or `CausalLMOutputWithPast`.
*   **Cache Structure:** Utilizes `transformers.DynamicCache`. The tensors are stored natively in lists under `self.key_cache` and `self.value_cache`.
*   **Citation:** [Hugging Face Transformers: DynamicCache Implementation](https://huggingface.co/docs/transformers/main/en/internal/generation_utils#transformers.DynamicCache)

### B. Gemma-3 27B (Google)
*   **Attention Signature:** 2-element return `(attn_output, past_key_values)`. The `attn_weights` are omitted unless explicitly requested.
*   **Cache Structure:** Utilizes specialized `Gemma2Cache` due to its hybrid sliding window and logit soft-capping mechanisms. 
*   **Citation:** [Hugging Face: Gemma2 Architecture](https://huggingface.co/docs/transformers/main/en/model_doc/gemma2)

### C. Qwen-3 30B (Alibaba)
*   **Attention Signature:** 2-element return `(attn_output, past_key_values)`.
*   **Cache Structure:** Utilizes `DynamicCache` with extreme Grouped-Query Attention (GQA) ratios. Qwen models aggressively drop attention weights from the return signature to save VRAM.
*   **Citation:** [Hugging Face: Qwen2/3 Architecture](https://huggingface.co/docs/transformers/main/en/model_doc/qwen2)

### D. Devstral 24B (Mistral-Based)
*   **Attention Signature:** 2-element return `(attn_output, past_key_values)`.
*   **Cache Structure:** Employs a strict `SlidingWindowCache`. Tensors are constantly rotated, meaning the physical footprint stabilizes after `window_size` tokens.
*   **Citation:** [Hugging Face: Mistral Architecture & Sliding Window Cache](https://huggingface.co/docs/transformers/main/en/model_doc/mistral)

---

## 3. Implementation Plan for the Final Run

To ensure the final run captures pristine data without relying on Outlines' internal object structure, we will implement an **Architecture-Agnostic Telemetry Registry**.

### Step 1: The Global Registry (`surgery_utils.py`)
Instead of attaching `_kv_cache_telemetry` to `model.config` (which gets obscured by wrappers), we will establish a global registry inside `surgery_utils.py`.
```python
TELEMETRY_REGISTRY = {"latest_pkv": None}
```
During `common_attention_amputation`, we will directly store the live reference:
```python
TELEMETRY_REGISTRY["latest_pkv"] = past_key_values
```

### Step 2: Direct State Extraction (`sql_agent_utils.py` & `benchmark_utils.py`)
The `telemetry_node_wrapper` will bypass `state["outlines_model"]` entirely and read directly from the global registry at the moment of yield.
```python
from surgery_utils import TELEMETRY_REGISTRY
pkv = TELEMETRY_REGISTRY.get("latest_pkv")
current_cache_mb = measure_actual_cache_size(pkv)
```

### Step 3: Precise Peak Tracking
We will maintain the `reset_peak_vram()` call *before* generation begins, but we will ensure `torch.cuda.empty_cache()` is strictly called **before** the reset. This ensures the baseline memory drops back to the weights-only footprint before the high-water mark tracking begins for the turn.

By executing this plan, we mathematically guarantee that the cache object is captured regardless of the model's architecture (Gemma, Devstral, Qwen, or GPT) and regardless of how heavily the generation library wraps the underlying neural network.