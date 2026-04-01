#!/bin/bash

# ==========================================
# THE HYPERCONDUCTOR MASTER LOOP (V3: High-Density Gauntlet)
# Dynamic Context & Multi-Architecture Integration
# ==========================================

# Path to the virtual environment python
VENV_PATH="python3"

# ---------------------------------------------------------
# PHASE 0: ENVIRONMENT SETUP (Colab/L4 Optimization)
# ---------------------------------------------------------
echo "[SYSTEM] Checking dependencies..."
$VENV_PATH -c "import bitsandbytes" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "[SYSTEM] bitsandbytes not found. Synchronizing environment..."
    # 1. Install core AI extensions
    $VENV_PATH -m pip install --no-cache-dir bitsandbytes>=0.46.1 accelerate>=0.29.0 peft transformers>=4.40.0
    
    # 2. Force fix the torchvision/torch CUDA mismatch
    echo "[SYSTEM] Fixing CUDA version synchronization..."
    $VENV_PATH -m pip install --no-cache-dir -U torchvision --index-url https://download.pytorch.org/whl/cu121 2>/dev/null || echo "[WARNING] Torchvision fix failed."
    
    echo "[SYSTEM] Dependencies installed."
else
    echo "[SYSTEM] Dependencies verified."
fi

# Define the model sequence (Hugging Face Hub IDs)
MODELS=(
    "Qwen/Qwen2.5-1.5B-Instruct"
    "microsoft/Phi-3-mini-4k-instruct"
    "google/gemma-2-2b-it"
    "meta-llama/Llama-3.2-3B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    "mistralai/Mistral-7B-Instruct-v0.3"
)

SAMPLE_TEXT="sample_texts/lorem_50k.txt"

echo "[SYSTEM] Initializing Telemetry Database..."
$VENV_PATH init_db.py

# Ensure the CSV exists with headers
if [ ! -f metrics.csv ]; then
    echo "model_name,lambda_b,run_id,max_chars,passkey_retrieval_acc,perplexity_score,peak_vram_mb,final_cache_size_mb,ttft_ms,tpot_ms" > metrics.csv
fi

echo "[SYSTEM] Commencing Comprehensive Evolutionary Gauntlet..."

for MODEL in "${MODELS[@]}"; do
    echo "=========================================="
    echo ">> TARGET MODEL: $MODEL"
    echo "=========================================="

    # --- DYNAMIC CONTEXT SCALING ---
    if [[ $MODEL == *"Qwen"* ]]; then 
        RUN_MAX_CHARS=10000
    elif [[ $MODEL == *"Phi"* ]]; then 
        RUN_MAX_CHARS=3000
    elif [[ $MODEL == *"gemma"* ]]; then 
        RUN_MAX_CHARS=5000
    elif [[ $MODEL == *"Llama-3.2-3B"* ]]; then 
        RUN_MAX_CHARS=12000
    elif [[ $MODEL == *"Llama-3.1-8B"* ]]; then 
        RUN_MAX_CHARS=16000
    elif [[ $MODEL == *"Mistral"* ]]; then 
        RUN_MAX_CHARS=16000
    else 
        RUN_MAX_CHARS=6000
    fi
    echo "[DYNAMIC] Scaling context window to $RUN_MAX_CHARS chars for $MODEL"

    # PHASE 1: BISECTION SEARCH (Find scalar Discontinuity Threshold)
    echo "[ORCHESTRATOR] Starting Bisection Search for scalar baseline..."
    $VENV_PATH bisection_search.py --model "$MODEL"
    
    # Extract the last successful lambda_b
    echo "[ORCHESTRATOR] Identifying scalar boundary..."
    LAMBDA_BASE=$(export TARGET_MODEL="$MODEL"; $VENV_PATH -c "import pandas as pd; import os; df = pd.read_csv('metrics.csv'); target = os.environ.get('TARGET_MODEL'); success = df[(df['model_name']==target) & (df['passkey_retrieval_acc']==1.0)]; print(success['lambda_b'].max() if not success.empty else '0.001')")
    
    # Safety check for NaN or empty
    if [[ "$LAMBDA_BASE" == "None" ]] || [[ -z "$LAMBDA_BASE" ]] || [[ "$LAMBDA_BASE" == "nan" ]]; then
        LAMBDA_BASE="0.001"
    fi
    echo "[ORCHESTRATOR] Scalar Boundary identified: $LAMBDA_BASE"

    # PHASE 2: ENTROPY CALIBRATION
    MATRIX_FILE="lambda_matrix_${MODEL//\//_}.pt"
    echo "[ORCHESTRATOR] Generating Entropy-Aware Penalty Matrix..."
    # We use a smaller context (4000 chars) for calibration to avoid OOM with raw attention matrices
    $VENV_PATH extract_entropy.py --model "$MODEL" --lambda_base "$LAMBDA_BASE" --output "$MATRIX_FILE" --text "$SAMPLE_TEXT" --max_chars 4000

    # PHASE 3: MULTIDIMENSIONAL EXECUTION
    echo "[ORCHESTRATOR] Evaluating Multidimensional Surgery..."
    $VENV_PATH run_benchmarks.py \
        --model "$MODEL" \
        --lambda_b "$LAMBDA_BASE" \
        --lambda_matrix_path "$MATRIX_FILE" \
        --max_chars "$RUN_MAX_CHARS" \
        --text "$SAMPLE_TEXT" \
        --run_id "multidimensional_v3"

    # CLEANUP
    echo "[CLEANUP] Purging VRAM and OS Caches..."
    pkill -f "run_benchmarks.py" || true
    sync; echo 3 2>/dev/null > /proc/sys/vm/drop_caches || true
    sleep 5 
done

# PHASE 4: TELEMETRY ANALYSIS & MAPPING
echo "[ORCHESTRATOR] Consolidating results..."
$VENV_PATH run_mapping.py

# PHASE 5: AUTO-TERMINATION (Colab Pro Efficiency)
echo "[ORCHESTRATOR] Gauntlet complete. Shutting down Colab instance..."
python3 -c "from google.colab import runtime; runtime.unassign()" || echo "[WARNING] Shutdown command failed. Manual termination required."

echo "=========================================="
echo "[SYSTEM] Comprehensive Gauntlet Complete."
echo "=========================================="
