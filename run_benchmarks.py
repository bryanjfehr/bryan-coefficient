#bryan-coefficient/run_benchmarks.py
import argparse
import sys
import torch
import time
import sqlite3
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from surgery_utils import patch_attention
from benchmark_utils import inject_passkey, verify_passkey, calculate_ppl, load_benchmark_dataset, persist_metrics
from profiler_utils import get_peak_vram, measure_generation_latency, measure_actual_cache_size

def apply_bryan_coefficient(model, lambda_b, lambda_b_matrix=None):
    """
    Surgically integrates λB eviction into the model's forward pass 
    using strict architecture routing.
    Supports both scalar and multidimensional penalties.
    """
    model_type = model.config.model_type
    if lambda_b_matrix is not None:
        print(f"[SURGERY] Detected Architecture: {model_type}. Applying Multidimensional λB,h Matrix...")
    else:
        print(f"[SURGERY] Detected Architecture: {model_type}. Applying λB = {lambda_b}...")
    
    patched_count = 0
    # Robustly discovery of layer count across diverse architectures
    num_layers = getattr(model.config, "num_hidden_layers", 
                         getattr(model.config, "num_layers", 
                                 getattr(model.config, "n_layers", None)))
    
    if num_layers is None:
        # Fallback: Count the number of items in model.layers or equivalent if possible
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            num_layers = len(model.model.layers)
        elif hasattr(model, "layers"):
            num_layers = len(model.layers)
        else:
            # Last resort: use a reasonable default or scan named modules once
            num_layers = 0
            for name, _ in model.named_modules():
                if "layers." in name:
                    import re
                    match = re.search(r'layers\.(\d+)', name)
                    if match:
                        num_layers = max(num_layers, int(match.group(1)) + 1)
    
    if num_layers == 0: num_layers = 32 # Final fallback
    
    from surgery_utils import patch_attention
    for name, module in model.named_modules():
        cls_name = str(type(module))
        # Robust discovery: look for modules that look like Attention layers
        is_attn = ("Attention" in cls_name or "attn" in name.lower()) and \
                  (hasattr(module, "q_proj") or hasattr(module, "qkv_proj"))
        
        if is_attn:
            layer_idx = getattr(module, "layer_idx", None)
            
            # If layer_idx is missing, try to extract it from the name (e.g. "model.layers.0.self_attn")
            if layer_idx is None:
                import re
                match = re.search(r'layers\.(\d+)\.', name)
                if match:
                    layer_idx = int(match.group(1))
            
            # DYNAMIC LAYER SCALING:
            # Shield first 20% and last 20% of layers.
            # Use a trapezoidal scaling: 0 at ends, 1 in middle.
            scale = 1.0
            if layer_idx is not None and num_layers > 5:
                # Shielding logic
                early_boundary = num_layers // 5
                late_boundary = num_layers - early_boundary
                
                if layer_idx < early_boundary:
                    scale = layer_idx / early_boundary
                elif layer_idx > late_boundary:
                    scale = (num_layers - 1 - layer_idx) / (num_layers - late_boundary)
            
            module.lb_scale = scale
            effective_lb = lambda_b * scale
            
            print(f"[DEBUG] Patching Module: {name} | Layer: {layer_idx} | Scaling λB: {effective_lb:.6f} (Scale: {scale:.2f})")
            patch_attention(module, lambda_b=effective_lb, lambda_b_matrix=lambda_b_matrix, layer_idx=layer_idx)
            patched_count += 1

    if patched_count == 0:
        print("[FATAL ERROR] Surgery failed! No compatible attention layers found.")
        sys.exit(1)
        
    print(f"[SURGERY] Successfully patched {patched_count} attention layers.")
    return model

def run_passkey_test(model, tokenizer, text_path, lambda_b, insert_idx=None, max_chars=8000):
    """
    Executes the Passkey Retrieval benchmark.
    """
    # 0. Conditional Truncation for Phi Models (The 4096-token context wall)
    if model.config.model_type == "phi3":
        if max_chars > 12000:
            print(f"[SYSTEM] Truncating Phi-3 input from {max_chars} to 12000 characters to prevent context overflow.")
            max_chars = 12000

    if insert_idx is None:
        insert_idx = max_chars // 2
    # 1. Load the Robert Greene Haystack
    try:
        with open(text_path, 'r', encoding='utf-8') as f:
            haystack = f.read()
    except FileNotFoundError:
        print(f"[ERROR] Sample text not found at {text_path}. Using fallback.")
        haystack = load_benchmark_dataset() # Use Wiki snippet as fallback

    # Truncate haystack to avoid OOM in Eager mode (8GB Limit)
    haystack = haystack[:max_chars]
    
    passkey = "84729"
    modified_text = inject_passkey(haystack, passkey, insert_idx)
    
    print(f"[DEBUG] Injection Point: ...{modified_text[insert_idx-25:insert_idx+50]}...")
    
    prompt = f"{modified_text}\n\nQuestion: What is the secret passkey? Answer:"

    # --- INTEGRITY CHECK ---
    from benchmark_utils import verify_prompt_integrity
    if not verify_prompt_integrity(prompt, passkey):
         print("[FATAL] Injection Script Failed. Passkey missing from raw prompt.")
         sys.exit(1)
    
    print(f"[DEBUG] Prompt Tail (50 chars): ...{prompt[-50:]}")
    # -----------------------

    # 3. Tokenization
    # We use truncation=True to handle cases where 12000 chars still might exceed 4096 tokens (unlikely but safe)
    max_pos = model.config.max_position_embeddings if hasattr(model.config, "max_position_embeddings") else 4096
    inputs = {k: v.to(model.device) for k, v in tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_pos).items()}
    
    # 4. Inference
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=20, 
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True
        )
    
    # Weigh the compressed cache
    cache_size_mb = measure_actual_cache_size(outputs.past_key_values)
    print(f"[METRIC] Final KV Cache Size: {cache_size_mb:.2f} MB")
    
    response = tokenizer.decode(outputs.sequences[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    success = 1.0 if verify_passkey(response, passkey) else 0.0
    
    return success, response, cache_size_mb

def run_ppl_test(model, tokenizer):
    """
    Executes the Perplexity (PPL) benchmark.
    """
    print(f"[BENCH] Starting Perplexity (PPL) Evaluation...")
    text = load_benchmark_dataset()
    
    # Force truncation to model's native context window
    max_pos = model.config.max_position_embeddings if hasattr(model.config, "max_position_embeddings") else 4096
    inputs = {k: v.to(model.device) for k, v in tokenizer(text, return_tensors="pt", truncation=True, max_length=max_pos).items()}
    
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        ppl = calculate_ppl(outputs.logits, inputs["input_ids"])
    
    print(f"[BENCH] Perplexity Score: {ppl:.4f}")
    return ppl

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--lambda_b", type=float, required=True)
    parser.add_argument("--text", type=str, default="sample_texts/rg_sample_text.txt")
    parser.add_argument("--db", type=str, default="lambda_b_telemetry.db")
    parser.add_argument("--max_chars", type=int, default=8000)
    parser.add_argument("--run_id", type=str, default="default")
    parser.add_argument("--no_surgery", action="store_true")
    parser.add_argument("--lambda_matrix_path", type=str, default=None, help="Path to the .pt lambda matrix")
    parser.add_argument("--agent_test", action="store_true", help="Run the agentic workflow ReAct test")
    parser.add_argument("--agent_db", type=str, default="agent_telemetry.db")
    args = parser.parse_args()

    # Hardware Setup: 4-bit Quantization for 16GB RAM
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True
    )
    
    print(f"[RUN] Evaluating λB = {args.lambda_b} on {args.model}")
    
    import os
    hf_token = os.environ.get("HF_TOKEN")

    from transformers import AutoConfig
    print(f"[SYSTEM] Checking model configuration for {args.model}...")
    config = AutoConfig.from_pretrained(args.model, token=hf_token, trust_remote_code=True)
    
    if hasattr(config, "quantization_config"):
        print(f"[SYSTEM] Model is pre-quantized. Skipping BitsAndBytes.")
        load_kwargs = {
            "device_map": "auto",
            "torch_dtype": "auto",
            "low_cpu_mem_usage": True,
            "attn_implementation": "eager",
            "token": hf_token,
            "trust_remote_code": True
        }
    else:
        print(f"[SYSTEM] Applying BitsAndBytes 4-bit quantization.")
        load_kwargs = {
            "quantization_config": bnb_config,
            "device_map": "auto",
            "low_cpu_mem_usage": True,
            "attn_implementation": "eager",
            "token": hf_token,
            "trust_remote_code": True
        }

    print(f"[SYSTEM] Loading Model: {args.model}")
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    except ValueError as e:
        if "Unrecognized configuration class" in str(e) and hasattr(config, "architectures"):
            arch_name = config.architectures[0]
            print(f"[SYSTEM] AutoModel recognition failure. Attempting direct load via {arch_name}...")
            import transformers
            arch_cls = getattr(transformers, arch_name, None)
            if arch_cls:
                model = arch_cls.from_pretrained(args.model, **load_kwargs)
            else:
                raise e
        else:
            raise e
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=hf_token, trust_remote_code=True)

    # Load multidimensional matrix if provided
    lambda_b_matrix = None
    if args.lambda_matrix_path and os.path.exists(args.lambda_matrix_path):
        print(f"[SYSTEM] Loading multidimensional λB matrix from {args.lambda_matrix_path}")
        lambda_b_matrix = torch.load(args.lambda_matrix_path)

    # Execute Surgery
    if not args.no_surgery:
        apply_bryan_coefficient(model, args.lambda_b, lambda_b_matrix=lambda_b_matrix)
    else:
        print("[SYSTEM] Skipping surgery (Baseline Run)")

    # --- THE GAUNTLET ---
    
    # 1. Hardware Baseline (Reset Stats)
    torch.cuda.reset_peak_memory_stats()
    
    if args.agent_test:
        print(f"[BENCH] Starting Agentic Workflow Test...")
        from sql_agent_utils import build_and_run_graph
        success, resp, _ = build_and_run_graph(model, tokenizer, args.lambda_b, args.run_id, db_name=args.agent_db)
        print(f"Agent Result: {'SUCCESS' if success == 1.0 else 'FAIL'} | Final: {resp}")
    else:
        # 2. Passkey Retrieval
        passkey_acc, resp, cache_size_mb = run_passkey_test(model, tokenizer, args.text, args.lambda_b, max_chars=args.max_chars)
        print(f"Result: {'SUCCESS' if passkey_acc == 1.0 else 'FAIL'} | Resp: {resp}")
        
        # 3. Perplexity
        ppl_score = run_ppl_test(model, tokenizer)
        
        # 4. Hardware Profiling
        peak_vram = get_peak_vram(unit="MB")
        ttft, tpot = measure_generation_latency(model, tokenizer, "The Bryan Coefficient is")
        
        # 5. Persistent Telemetry
        metrics = {
            "model_name": args.model,
            "lambda_b": args.lambda_b,
            "run_id": args.run_id,
            "max_chars": args.max_chars,
            "passkey_acc": passkey_acc,
            "model_response": resp,
            "ppl": ppl_score,
            "peak_vram_mb": peak_vram,
            "final_cache_size_mb": cache_size_mb,
            "ttft_ms": ttft,
            "tpot_ms": tpot,
            "cache_retention_pct": 1.0 # Placeholder
        }
        
        persist_metrics(args.db, metrics)
    
    print(f"[SUCCESS] Gauntlet complete for {args.model}")

if __name__ == "__main__":
    main()
