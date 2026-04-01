# Architecture Research: Bryan Coefficient Telemetry (A100/v4 Protocol)

## 1. Model Analysis
| Model | Family | Model Type | Expected Return Len | Cache Class |
| :--- | :--- | :--- | :--- | :--- |
| GPT-OSS 20B | GPT | `gpt_oss` | 3 (Standard) | `DynamicCache` |
| Gemma-3 27B | Gemma | `gemma2` / `gemma` | 2 | `Gemma2Cache` |
| Qwen-3 30B | Qwen | `qwen2` | 2 | `DynamicCache` |
| Devstral 24B | Mistral | `mistral` / `devstral` | 2 | `SlidingWindowCache` |

## 2. Identified Telemetry Gaps
*   **Mistral/Devstral Crash:** Likely due to `_return_len` mismatch if `model_type` is "devstral" but not "mistral".
*   **Static VRAM:** `torch.cuda.max_memory_allocated()` needs to be reset at the start of each turn to capture *turn-specific peak*, otherwise it just shows the global high-water mark.
*   **Pruning Ratio 0s:** `_pruning_history` might not be propagating if `self.config` on the module isn't the same object as `model.config`.
*   **Gemma/Devstral 0s Cache:** Modern transformers `Cache` objects (Gemma2/SlidingWindow) store tensors in `key_cache` and `value_cache` which might be nested or named differently in specialized subclasses.

## 3. Implementation Plan
### Surgery Updates (`surgery_utils.py`)
- Expand `_return_len` detection to include `devstral` and handle `gemma2`.
- Ensure `past_key_values` is captured via `model.config._kv_cache_telemetry` even if the model returns a complex object.
- Force `_pruning_history` to reside on the shared `model.config` if possible.

### Profiler Updates (`profiler_utils.py`)
- Add `reset_peak_vram()` to clear stats across all GPUs.
- Enhance `measure_actual_cache_size` with recursive discovery for `SlidingWindowCache`.

### Workflow Updates (`sql_agent_utils.py` / `benchmark_utils.py`)
- Call `reset_peak_vram()` before `outlines_model()` call.
- Update `telemetry_node_wrapper` to capture `get_peak_vram()` immediately after the yield.
