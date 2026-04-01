#!/bin/bash

# ==========================================
# THE AGENTIC GAUNTLET (V4: Outlines Hardened)
# Testing Reasoning Integrity Under KV Decay
# ==========================================

VENV_PATH="python3"

# ---------------------------------------------------------
# PHASE 0: ENVIRONMENT SETUP
# ---------------------------------------------------------
echo "[SYSTEM] Checking dependencies..."
$VENV_PATH -c "import outlines" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "[SYSTEM] Installing Outlines and Pydantic..."
    $VENV_PATH -m pip install --no-cache-dir outlines==1.2.12 pydantic>=2.12.0 bitsandbytes>=0.46.1 accelerate>=0.29.0 peft transformers>=4.40.0 langgraph langchain-core
    $VENV_PATH -m pip install --no-cache-dir -U torchvision --index-url https://download.pytorch.org/whl/cu121 2>/dev/null
    echo "[SYSTEM] Dependencies installed."
fi

# Ensure newly installed packages are in path for current session
export PYTHONPATH=$PYTHONPATH:$(python3 -m site --user-site)

# Environment Setup
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/agent_results_$TIMESTAMP"
mkdir -p "$RESULTS_DIR"
AGENT_DB="$RESULTS_DIR/agent_telemetry.db"

echo "[SYSTEM] Initializing Agent Telemetry at $AGENT_DB..."
$VENV_PATH init_agent_db.py --db "$AGENT_DB"

echo "[SYSTEM] Initializing SQL Live Environment..."
$VENV_PATH sql_eval_setup.py

# Models and their established Optimal Scalar Lambdas
MODELS=(
    "Qwen/Qwen2.5-1.5B-Instruct:0.0011"
    "microsoft/Phi-3-mini-4k-instruct:0.0025"
    "google/gemma-2-2b-it:0.0013"
    "meta-llama/Llama-3.2-3B-Instruct:0.0010"
    "meta-llama/Llama-3.1-8B-Instruct:0.0050"
    "mistralai/Mistral-7B-Instruct-v0.3:0.0013"
)

echo "[SYSTEM] Commencing Hardened Agentic Workflow Sweep..."

for ENTRY in "${MODELS[@]}"; do
    MODEL="${ENTRY%%:*}"
    LB="${ENTRY#*:}"
    
    echo "=========================================="
    echo ">> TARGET MODEL: $MODEL"
    echo "=========================================="

    # --- PHASE 1: PRE-FLIGHT BASELINE CHECK ---
    echo "[ORCHESTRATOR] Performing Pre-flight Baseline Check (λB = 0.0, Noise Level 2)..."
    $VENV_PATH run_benchmarks.py \
        --model "$MODEL" \
        --lambda_b 0.0 \
        --agent_test \
        --agent_db "$AGENT_DB" \
        --run_id "baseline_check_${MODEL//\//_}_$TIMESTAMP"
    
    # Check if the baseline succeeded (success=1.0 in the DB)
    BASELINE_SUCCESS=$(export TARGET_MODEL="$MODEL"; $VENV_PATH -c "import pandas as pd; import sqlite3; conn = sqlite3.connect('$AGENT_DB'); df = pd.read_sql_query('SELECT success FROM agent_runs WHERE model_name=\"' + '$MODEL' + '\" AND lambda_b=0.0', conn); print(df['success'].max() if not df.empty else '0.0')")
    
    if [[ "$BASELINE_SUCCESS" != "1.0" ]]; then
        echo "[FATAL ERROR] Model $MODEL failed baseline reasoning. Marking as Incapable and skipping gauntlet."
        continue
    fi
    echo "[ORCHESTRATOR] Baseline reasoning verified. Proceeding to KV Decay test."

    # --- PHASE 2: KV DECAY STRESS TEST ---
    MATRIX_FILE="lambda_matrix_${MODEL//\//_}.pt"
    EXTRA_ARGS=""
    if [ -f "$MATRIX_FILE" ]; then
        echo "[ORCHESTRATOR] Applying multidimensional head-aware surgery."
        EXTRA_ARGS="--lambda_matrix_path $MATRIX_FILE"
    fi

    echo "[ORCHESTRATOR] Executing Hardened Stress Test (λB = $LB)..."
    $VENV_PATH run_benchmarks.py \
        --model "$MODEL" \
        --lambda_b "$LB" \
        $EXTRA_ARGS \
        --agent_test \
        --agent_db "$AGENT_DB" \
        --run_id "agent_eval_${MODEL//\//_}_$TIMESTAMP"

    # CLEANUP
    pkill -f "run_benchmarks.py" || true
    sync; echo 3 2>/dev/null > /proc/sys/vm/drop_caches || true
    sleep 5 
done

# PHASE 4: DATA EXPORT
echo "[ORCHESTRATOR] Exporting results..."
$VENV_PATH export_agent_results.py --db "$AGENT_DB" --out "$RESULTS_DIR"

LATEST_LINK="results/latest_agent"
rm -f "$LATEST_LINK"
ln -s "$(pwd)/$RESULTS_DIR" "$LATEST_LINK"

echo "=========================================="
echo "[SYSTEM] Hardened Agentic Gauntlet Complete."
echo "[SYSTEM] Shutting down Colab instance..."
python3 -c "from google.colab import runtime; runtime.unassign()" || echo "[WARNING] Shutdown command failed."
