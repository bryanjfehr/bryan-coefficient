import sqlite3
import argparse

def init_agent_db(db_name="agent_telemetry.db"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    # Table for high-level agent run metrics
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS agent_runs (
            run_id TEXT PRIMARY KEY,
            model_name TEXT,
            lambda_b REAL,
            success REAL,
            total_turns INTEGER,
            failure_turn INTEGER,
            final_response TEXT,
            peak_vram_mb REAL,
            avg_cache_mb REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Table for granular conversation logging
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS agent_conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            turn_number INTEGER,
            agent_action TEXT,
            scratchpad TEXT,
            observation TEXT,
            vram_mb REAL,
            kv_cache_mb REAL,
            pruning_ratio REAL,
            lambda_b REAL,
            FOREIGN KEY(run_id) REFERENCES agent_runs(run_id)
        )
    ''')

    # Standard Intelligence Metrics (for Bisection/PPL)
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

    # Standard Hardware Metrics
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
    print(f"[SYSTEM] Unified Telemetry Database '{db_name}' initialized.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=str, default="agent_telemetry.db")
    args = parser.parse_args()
    init_agent_db(args.db)
