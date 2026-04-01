import subprocess
import sqlite3
import pandas as pd
import json
import os
import time
import argparse

# --- CONFIGURATION ---
STATE_FILE = "bisection_state.json"
METRICS_CSV = "metrics.csv"
TOLERANCE = 0.00005 # The floating-point precision to stop the search
SAMPLE_TEXT = "sample_texts/lorem_50k.txt"

def load_state(model_name):
    """Loads the last known bounds for a specific model."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            try:
                state = json.load(f)
                if model_name in state:
                    return state[model_name]["low"], state[model_name]["high"]
            except json.JSONDecodeError:
                pass
    
    # Default bounds if starting fresh
    return 0.0001, 0.01

def save_state(model_name, low, high):
    """Persists the search state."""
    state = {}
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            try:
                state = json.load(f)
            except json.JSONDecodeError:
                pass
            
    state[model_name] = {"low": low, "high": high}
    
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=4)

def get_max_chars(model_name):
    """Dynamic context scaling mirroring run_experiments.sh"""
    if "Qwen" in model_name: 
        return 10000
    elif "Phi" in model_name: 
        return 3000
    elif "gemma" in model_name: 
        return 5000
    elif "Llama-3.2-3B" in model_name: 
        return 12000
    elif "Llama-3.1-8B" in model_name: 
        return 16000
    elif "Mistral" in model_name: 
        return 16000
    else: 
        return 6000

def run_gauntlet_subprocess(model_name, lambda_b, db_path="agent_telemetry.db"):
    """
    Executes the benchmark script in a separate process.
    """
    max_chars = get_max_chars(model_name)
    print(f"\n[BISECTION] Testing {model_name} at λB = {lambda_b:.6f} (Context: {max_chars} chars)")
    
    cmd = [
        "python3", "run_benchmarks.py",
        "--model", model_name,
        "--lambda_b", str(lambda_b),
        "--max_chars", str(max_chars),
        "--text", SAMPLE_TEXT,
        "--run_id", "bisection_sweep",
        "--db", db_path
    ]
    
    subprocess.run(cmd, check=True)
    time.sleep(3)

def evaluate_run(model_name, lambda_b):
    """
    Reads the output metrics to determine if the network maintained
    structural integrity (Pass) or hit the amnesia cliff (Fail).
    """
    if not os.path.exists(METRICS_CSV):
        return False

    df = pd.read_csv(METRICS_CSV)
    
    # Filter for the exact run we just completed
    mask = (df['model_name'] == model_name) & (abs(df['lambda_b'] - lambda_b) < 1e-7)
    if not mask.any():
        return False
        
    recent_run = df[mask].iloc[-1]
    
    passkey_acc = recent_run['passkey_retrieval_acc']
    ppl_score = recent_run['perplexity_score']
    
    # --- EVALUATION CRITERIA ---
    if passkey_acc == 1.0 and ppl_score < 10.0:
        return True # The network held together
    else:
        return False # The network shattered (Cliff found)

def init_metrics_file():
    if not os.path.exists(METRICS_CSV):
        print("[SYSTEM] Initializing metrics.csv for Bisection Search...")
        with open(METRICS_CSV, "w") as f:
            f.write("model_name,lambda_b,run_id,max_chars,passkey_retrieval_acc,perplexity_score,peak_vram_mb,final_cache_size_mb,ttft_ms,tpot_ms\n")

def find_discontinuity_cliff(target_model=None, db_path="agent_telemetry.db"):
    """The core binary bisection loop."""
    models_to_test = [target_model] if target_model else [
        "Qwen/Qwen2.5-1.5B-Instruct",
        "microsoft/Phi-3-mini-4k-instruct",
        "google/gemma-2-2b-it",
        "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3"
    ]
    
    for model in models_to_test:
        print(f"\n==================================================")
        print(f"[ORCHESTRATOR] Mapping Discontinuity Cliff for {model}")
        print(f"==================================================")
        
        lambda_low, lambda_high = load_state(model)
        
        while (lambda_high - lambda_low) > TOLERANCE:
            lambda_mid = (lambda_low + lambda_high) / 2.0
            
            try:
                run_gauntlet_subprocess(model, lambda_mid, db_path=db_path)
                passed = evaluate_run(model, lambda_mid)
                
                if passed:
                    print(f"[RESULT] SUCCESS at {lambda_mid:.6f}. Pushing higher...")
                    lambda_low = lambda_mid
                else:
                    print(f"[RESULT] FAILURE at {lambda_mid:.6f}. Pulling back...")
                    lambda_high = lambda_mid
                
                save_state(model, lambda_low, lambda_high)
                
            except subprocess.CalledProcessError:
                print(f"[FATAL ERROR] run_benchmarks.py crashed.")
                break
                
        cliff_value = lambda_low
        print(f"\n[MAPPED] Final Bryan Relation Boundary for {model}: {cliff_value:.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="Target model ID")
    parser.add_argument("--db", type=str, default="agent_telemetry.db", help="Path to telemetry DB")
    args = parser.parse_args()
    
    init_metrics_file()
    find_discontinuity_cliff(target_model=args.model, db_path=args.db)
