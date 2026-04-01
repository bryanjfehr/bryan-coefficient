#!/bin/bash
# repro_large_agent.sh: Orchestration for Large Model Evaluation (A100)
# This script executes the full λB protocol for 8B+ models.

# 1. Environment Sync
echo "[SYSTEM] Synchronizing high-compute dependencies..."
pip install -q transformers bitsandbytes accelerate langgraph langchain-core pandas outlines

# 2. Models to Evaluate (Tier 3: Large Models)
# These models require an A100 to avoid OOM in Eager mode with long context.
MODELS=(
    "meta-llama/Llama-3.1-8B-Instruct"
    "mistralai/Mistral-7B-Instruct-v0.3"
)

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPRO_DB="repro_large_agent_$TIMESTAMP.db"

# 3. Evaluation Loop
for MODEL in "${MODELS[@]}"; do
    echo "=========================================="
    echo ">> TARGET MODEL: $MODEL (A100 Tier)"
    echo "=========================================="

    # Phase 1: Bisection Search (Find the Discontinuity Cliff)
    echo "[REPRO] Starting Bisection Search..."
    python3 bisection_search.py --model "$MODEL" --db "repro_metrics_$TIMESTAMP.db"

    # Identify the boundary from metrics
    LB=$(export TARGET_MODEL="$MODEL"; python3 -c "import pandas as pd; import sqlite3; conn = sqlite3.connect('repro_metrics_$TIMESTAMP.db'); df = pd.read_sql_query('SELECT lambda_b, passkey_retrieval_acc FROM metrics WHERE model_name=\"' + '$MODEL' + '\"', conn); print(df[df['passkey_retrieval_acc']==1.0]['lambda_b'].max() if not df[df['passkey_retrieval_acc']==1.0].empty else '0.001')")

    # Phase 2: Agentic Stress Test at the Cliff
    echo "[REPRO] Identified Boundary λB: $LB"
    echo "[REPRO] Executing Agentic Stress Test..."
    
    python3 run_benchmarks.py \
        --model "$MODEL" \
        --lambda_b "$LB" \
        --agent_test \
        --agent_db "$REPRO_DB" \
        --run_id "repro_large_${MODEL//\//_}"

    # Cleanup VRAM
    pkill -f "run_benchmarks.py" || true
    sleep 5
done

echo "[SUCCESS] Large-scale evaluation reproduction complete."
echo "[INFO] Results saved to repro_metrics_$TIMESTAMP.db and $REPRO_DB"
