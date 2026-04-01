# Post-Mortem Analysis: large_gauntlet_20260331_201018

## 1. The Catastrophic Failure: Gemma-3-27B
The run crashed instantly for all Gemma-3 lambda evaluations with: `LangGraph Execution Crash: argument of type 'NoneType' is not iterable`.

**Root Cause:**
In `sql_agent_utils.py`, the `safe_model_call` handles the `Outlines` generation.
When `lambda_b > 0`, the model's KV cache is heavily pruned. If it's pruned too aggressively or improperly padded, the model generates garbage tokens or `None` during constrained generation. 
The LangGraph loop expects a valid JSON string to parse into messages. When `safe_model_call` returned `None` or an empty string `""`, it was added to the state array as a message with `content=""` and potentially a missing or `None` role.
In the subsequent turn, `reasoning_node` executes:
```python
role = msg.get("role", "user").upper()
```
Because the failed message generation resulted in `msg.get("role")` being explicitly `None` (rather than omitted), the `.upper()` call threw an AttributeError.

## 2. Telemetry Progress & Remaining Issues
*   **GPT-OSS & Qwen3 & Devstral:** The new `TELEMETRY_REGISTRY` successfully captured `peak_vram_mb` and `avg_cache_mb` (except for Devstral cache, which still returned 0.0).
*   **Devstral's 0.0 Cache:** The `mistralai/Devstral` model utilizes a `SlidingWindowCache`. In `transformers` v5.0+ (the default on Colab A100s in 2026), the `SlidingWindowCache` structure does not always expose its internal tensors directly through `__dict__` or `__slots__` due to C++ extensions or optimized Triton backend integrations on A100s. We need to explicitly check for `.get_seq_length()` and use the specific API methods `[layer_idx][0]` and `[layer_idx][1]` to extract the tuple of key/value tensors if `key_cache` is hidden.

## 3. Targeted Fix Plan

### Fix 1: Robust State Handling (Already Applied)
I have updated `reasoning_node` and `action_node` in `sql_agent_utils.py` and `benchmark_utils.py` to check `if role is None: role = "user"` before calling `.upper()`. This prevents the entire gauntlet from crashing when a model hallucinates under extreme $\lambda_B$ pressure.

### Fix 2: Devstral / Transformers v5 Cache Weighing
I will update `measure_actual_cache_size` to handle opaque cache objects by leveraging the standard `transformers` tuple extraction fallback, ensuring Devstral's `SlidingWindowCache` is accurately weighed.

### Fix 3: Ensure v4_benchmark.py logs avg_cache
The user noted that `v4_benchmark.py` in the `bryan-coefficient` folder doesn't log `avg_cache_mb` to the CSV, which is true. It only logs `peak_vram_mb`. I will update `init_csv_logs` and the logging block to include `avg_cache_mb`.