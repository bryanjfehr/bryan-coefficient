import sqlite3
import pandas as pd
import argparse
import os

def export_agent_data(db_path, output_dir):
    if not os.path.exists(db_path):
        print(f"[ERROR] Database not found at {db_path}")
        return

    conn = sqlite3.connect(db_path)
    
    # 1. Export Agent Runs (High-level metrics)
    print(f"[SYSTEM] Exporting agent_runs from {db_path}...")
    df_runs = pd.read_sql_query("SELECT * FROM agent_runs", conn)
    runs_csv = os.path.join(output_dir, "agent_runs.csv")
    df_runs.to_csv(runs_csv, index=False)
    print(f"[SUCCESS] Agent runs saved to {runs_csv}")

    # 2. Export Agent Conversations (Granular logs)
    print(f"[SYSTEM] Exporting agent_conversations...")
    df_convs = pd.read_sql_query("SELECT * FROM agent_conversations", conn)
    convs_csv = os.path.join(output_dir, "agent_conversations.csv")
    df_convs.to_csv(convs_csv, index=False)
    print(f"[SUCCESS] Agent conversations saved to {convs_csv}")

    conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=str, required=True, help="Path to agent_telemetry.db")
    parser.add_argument("--out", type=str, required=True, help="Directory to save CSVs")
    args = parser.parse_args()
    
    export_agent_data(args.db, args.out)
