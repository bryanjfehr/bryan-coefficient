import torch
import time

def get_peak_vram(unit="MB", reset=False):
    """
    Interfaces with torch.cuda to capture Peak VRAM allocation across all devices.
    """
    if not torch.cuda.is_available():
        return 0.0
        
    total_peak_bytes = 0
    for i in range(torch.cuda.device_count()):
        total_peak_bytes += torch.cuda.max_memory_allocated(i)
        if reset:
            torch.cuda.reset_peak_memory_stats(i)
        
    if unit == "GB":
        return total_peak_bytes / (1024 ** 3)
    else: # Default to MB
        return total_peak_bytes / (1024 ** 2)

def reset_peak_vram():
    """
    Clears peak memory statistics across all available CUDA devices.
    """
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)

def get_current_vram(unit="MB"):
    """
    Returns the currently allocated VRAM across all available devices.
    """
    if not torch.cuda.is_available():
        return 0.0
    
    total_curr_bytes = 0
    for i in range(torch.cuda.device_count()):
        total_curr_bytes += torch.cuda.memory_allocated(i)
        
    if unit == "GB":
        return total_curr_bytes / (1024 ** 3)
    else:
        return total_curr_bytes / (1024 ** 2)

def measure_actual_cache_size(past_key_values):
    """
    Physically weighs the KV cache in Megabytes.
    Academic fidelity: Recursively traverses any nested structure to find all unique tensors.
    Handles __slots__, custom classes, and prevents infinite loops.
    """
    total_bytes = 0
    if past_key_values is None:
        return 0.0

    seen_ids = set()
    seen_tensors = set()

    def _weigh_recursive(obj):
        nonlocal total_bytes
        
        # Prevent infinite recursion for circular references
        obj_id = id(obj)
        if obj_id in seen_ids:
            return
        seen_ids.add(obj_id)

        if torch.is_tensor(obj):
            # Track physical memory pointers to avoid double-counting views/slices
            ptr = obj.data_ptr()
            if ptr not in seen_tensors:
                total_bytes += obj.element_size() * obj.nelement()
                seen_tensors.add(ptr)
        
        elif isinstance(obj, (list, tuple, set)):
            for item in obj:
                _weigh_recursive(item)
        
        elif isinstance(obj, dict):
            for v in obj.values():
                _weigh_recursive(v)
        
        elif hasattr(obj, "__dict__") or hasattr(obj, "__slots__") or hasattr(obj, "key_cache"):
            # Aggressive discovery: check dict, slots, and dir()
            attrs = set()
            if hasattr(obj, "__dict__"): attrs.update(obj.__dict__.keys())
            if hasattr(obj, "__slots__"): attrs.update(obj.__slots__)
            # Include common Transformers attribute names if not already found
            for common in ["key_cache", "value_cache", "keys", "values", "_key_cache", "_value_cache", "layers"]:
                if hasattr(obj, common): attrs.add(common)
            
            for attr in attrs:
                try:
                    val = getattr(obj, attr)
                    if not callable(val):
                        _weigh_recursive(val)
                except:
                    pass

        # Fallback for Transformers v5+ Opaque Caches (e.g. SlidingWindowCache via Triton)
        # These sometimes hide tensors behind the __iter__ or [layer_idx] tuple interface
        if hasattr(obj, "__iter__") and not isinstance(obj, dict) and not torch.is_tensor(obj):
            try:
                for item in obj:
                    _weigh_recursive(item)
            except:
                pass

    _weigh_recursive(past_key_values)
    return total_bytes / (1024 * 1024) # Return in MB

def measure_generation_latency(model, tokenizer, prompt, max_new_tokens=20):
    """
    Measures Time To First Token (TTFT) and Time Per Output Token (TPOT).
    Returns (ttft_ms, tpot_ms).
    """
    # Tokenize and move to device
    inputs = {k: v.to(model.device) for k, v in tokenizer(prompt, return_tensors="pt").items()}
    
    # 1. Measure TTFT (generating exactly 1 new token)
    start_ttft = time.time()
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=1, do_sample=False, use_cache=True)
    ttft_ms = (time.time() - start_ttft) * 1000
    
    # 2. Measure TPOT (generating N tokens and subtracting TTFT)
    if max_new_tokens > 1:
        start_tpot = time.time()
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
        total_ms = (time.time() - start_tpot) * 1000
        
        # Approximate TPOT: (Total - TTFT) / (Remaining Tokens)
        tpot_ms = (total_ms - ttft_ms) / (max_new_tokens - 1)
    else:
        tpot_ms = 0.0
        
    return ttft_ms, tpot_ms
