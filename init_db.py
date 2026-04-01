import sqlite3
import os

def initialize_database(db_name="lambda_b_telemetry.db"):
    # Do NOT remove existing database to allow cross-epoch tracking
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Table 1: Intelligence Metrics
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Intelligence_Metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        epoch INTEGER DEFAULT 1,
        run_id TEXT,
        model_name TEXT,
        lambda_b REAL,
        max_chars INTEGER,
        passkey_retrieval_acc REAL,
        perplexity_score REAL,
        model_response TEXT
    )
    ''')

    # Table 2: Hardware & Profiling Metrics
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Hardware_Metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        epoch INTEGER DEFAULT 1,
        run_id TEXT,
        model_name TEXT,
        lambda_b REAL,
        max_chars INTEGER,
        peak_vram_mb REAL,
        final_cache_size_mb REAL,
        ttft_ms REAL,
        tpot_ms REAL,
        cache_retention_pct REAL
    )
    ''')

    conn.commit()
    conn.close()
    print(f"[SYSTEM] Telemetry Database '{db_name}' initialized successfully.")

if __name__ == "__main__":
    initialize_database()