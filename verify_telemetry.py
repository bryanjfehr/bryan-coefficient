import sqlite3
import os
from benchmark_utils import persist_metrics
from init_db import initialize_database

def main():
    print("--- [PHASE 4: TELEMETRY PERSISTENCE VERIFICATION] ---")
    db_name = "lambda_b_telemetry.db"
    if os.path.exists(db_name):
        os.remove(db_name)
    initialize_database(db_name)

    metrics = {
        "model_name": "Qwen2-1.5B",
        "lambda_b": 0.01,
        "passkey_acc": 1.0,
        "ppl": 10.0,
        "peak_vram_mb": 500.0,
        "ttft_ms": 25.0,
        "tpot_ms": 15.0,
        "cache_retention_pct": 0.9
    }

    persist_metrics(db_name, metrics)

    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    print("\n[DATABASE VERIFICATION]")
    cursor.execute("SELECT * FROM Intelligence_Metrics")
    intel_row = cursor.fetchone()
    print(f"Intelligence Row: {intel_row}")

    cursor.execute("SELECT * FROM Hardware_Metrics")
    hw_row = cursor.fetchone()
    print(f"Hardware Row: {hw_row}")
    conn.close()

    if intel_row and hw_row and intel_row[2] == "Qwen2-1.5B":
        print("\n[SUCCESS] Telemetry persistence verified.")
    else:
        print("\n[FAILURE] Telemetry persistence verification failed.")

if __name__ == "__main__":
    main()
