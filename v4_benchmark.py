# v4_benchmark.py
import argparse
import sys
import torch
import os
import sqlite3
import time
import csv
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sql_agent_utils import build_and_run_graph
from run_benchmarks import apply_bryan_coefficient
from sql_eval_setup import setup_agent_database
from profiler_utils import get_peak_vram

RESULTS_FILE = "v4_agent_runs_progressive.csv"
CONVERSATION_LOG_FILE = "v4_agent_conversations.csv"

def init_csv_logs():
    """Initializes the CSV files with the appropriate headers."""
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                "run_id", "model_name", "lambda_b", "success", "total_turns", 
                "failure_mode", "peak_vram_mb", "avg_cache_mb", "timestamp"
            ])
            
    if not os.path.exists(CONVERSATION_LOG_FILE):
        with open(CONVERSATION_LOG_FILE, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                "run_id", "turn_number", "node_type", "content"
            ])

def analyze_failure_mode(state: dict) -> str:
    """Diagnoses exactly how the coefficient broke the reasoning."""
    if state.get("task_completed"):
        return "None"
    
    last_message = state.get("messages", [{}])[-1].get("content", "")
    
    if "Critical syntax error in JSON" in last_message or "ERROR_PARSING" in last_message:
        return "Syntax Fracture (Failed JSON formatting)"
    if "SQL Syntax Error" in last_message:
        return "Schema Hallucination (Invalid SQL query)"
    if state.get("current_step", 0) >= 10:
        return "Contextual Amnesia (Exceeded Max Turns / Looping)"
    
    return "Unknown Failure"

def main():
    parser = argparse.ArgumentParser(description="AGENTEVAL v4: Progressive Evaluation of Bryan Coefficient")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model ID")
    parser.add_argument("--lambdas", type=str, default="0.0,0.0005,0.001,0.0015,0.002", help="Comma-separated list of lambda_b values to test")
    parser.add_argument("--db", type=str, default="agent_telemetry.db", help="Path to telemetry database")
    parser.add_argument("--sql_db", type=str, default="multai_agent_eval.db", help="Path to live SQL environment")
    parser.add_argument("--max_turns", type=int, default=10, help="Max turns for the state machine")
    args = parser.parse_args()

    # Initialize CSV Logs
    init_csv_logs()

    # 1. Setup Environment
    print(f"[SYSTEM] Initializing SQL Live Environment at {args.sql_db}...")
    setup_agent_database(args.sql_db)

    lambda_list = [float(lb) for lb in args.lambdas.split(",")]
    hf_token = os.environ.get("HF_TOKEN")
    
    # 2. Hardware Setup (4-bit Quantization)
    from transformers import AutoConfig
    print(f"[SYSTEM] Checking model configuration for {args.model}...")
    config = AutoConfig.from_pretrained(args.model, token=hf_token, trust_remote_code=True)
    
    # Determine if we should apply our own BitsAndBytes config
    # If the model is already quantized (AWQ, GPTQ, MXFP4), we must NOT pass BitsAndBytesConfig
    if hasattr(config, "quantization_config"):
        print(f"[SYSTEM] Model is pre-quantized with {config.quantization_config.get('quant_method', 'unknown method')}. Skipping BitsAndBytes.")
        load_kwargs = {
            "device_map": "auto",
            "torch_dtype": "auto",
            "low_cpu_mem_usage": True,
            "attn_implementation": "eager",
            "token": hf_token,
            "trust_remote_code": True
        }
    else:
        print(f"[SYSTEM] Model is unquantized. Applying BitsAndBytes 4-bit quantization.")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True
        )
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

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 3. Progressive Evaluation Sweep
    patched = False
    for lb in lambda_list:
        run_id = f"v4_{args.model.split('/')[-1]}_L{lb}_{timestamp}"
        print(f"\n" + "="*60)
        print(f">> COMMENCING EVALUATION: λB = {lb}")
        print("="*60)

        # Apply or Update surgery
        if not patched:
            print(f"[SURGERY] First-time patching model with λB = {lb}")
            apply_bryan_coefficient(model, lb)
            patched = True
        else:
            print(f"[SURGERY] Updating existing patches to λB = {lb}")
            # Efficient update: set the attribute on all patched modules
            updated_count = 0
            for name, module in model.named_modules():
                if hasattr(module, "lambda_b"):
                    scale = getattr(module, "lb_scale", 1.0)
                    module.lambda_b = lb * scale
                    updated_count += 1
            print(f"[SURGERY] Updated λB on {updated_count} layers with dynamic scaling.")

        # Execute Gauntlet
        success, final_resp, final_state = build_and_run_graph(
            model, 
            tokenizer, 
            lb, 
            run_id, 
            db_name=args.db, 
            max_turns=args.max_turns
        )

        print(f"\n[RESULT] λB = {lb} | Success: {success} | Final: {final_resp}")
        
        # Analyze Failure Mode
        failure_mode = analyze_failure_mode(final_state)
        
        # 1. Log the High-Level Result to CSV
        peak_vram = get_peak_vram()
        kv_cache_history = final_state.get("kv_cache_history", [])
        avg_cache_mb = sum(kv_cache_history) / len(kv_cache_history) if kv_cache_history else 0.0
        
        with open(RESULTS_FILE, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                run_id, args.model, lb, success, final_state.get("current_step", 0),
                failure_mode, peak_vram, avg_cache_mb, datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ])
            
        # 2. Log the Turn-by-Turn Conversation to CSV
        with open(CONVERSATION_LOG_FILE, mode='a', newline='') as file:
            writer = csv.writer(file)
            for i, msg in enumerate(final_state.get("messages", [])):
                writer.writerow([run_id, i+1, msg.get("role"), msg.get("content")])
        
        # --- Progressive Logging (No longer breaking on failure to permit full observation) ---
        if success != 1.0:
            print(f"  [!] Task failed at λ_B = {lb}. Failure Mode: {failure_mode}")
            if lb == 0.0:
                print(f"  [!] Warning: Baseline failure detected. Continuing with sweep for VRAM/Response observation.")
        else:
            print(f"  [SUCCESS] Task completed at λ_B = {lb}")
        
        # Memory Cleanup between runs
        torch.cuda.empty_cache()
        import gc
        gc.collect()

    print("\n[SYSTEM] Progressive Evaluation Complete.")
    print(f"Results saved to {RESULTS_FILE} and {CONVERSATION_LOG_FILE}")

if __name__ == "__main__":
    main()
