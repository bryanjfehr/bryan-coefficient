import sqlite3
import pandas as pd
import os
import numpy as np

def fetch_metrics(db_path="lambda_b_telemetry.db"):
    """
    Fetches intelligence and hardware metrics from the SQLite database and joins them.
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at {db_path}")

    conn = sqlite3.connect(db_path)
    
    # Read tables into DataFrames
    query_intel = "SELECT * FROM Intelligence_Metrics"
    query_hw = "SELECT * FROM Hardware_Metrics"
    
    df_intel = pd.read_sql_query(query_intel, conn)
    df_hw = pd.read_sql_query(query_hw, conn)
    
    conn.close()

    if df_intel.empty or df_hw.empty:
        return pd.DataFrame()

    # Aggregate duplicates before joining to avoid Cartesian product
    # We take the mean of numeric columns, but keep model_name, lambda_b, epoch, run_id, max_chars as keys
    group_keys = ['model_name', 'lambda_b', 'epoch', 'run_id', 'max_chars']
    # Filter group_keys to only those that exist in the dataframe (handles old data)
    intel_keys = [k for k in group_keys if k in df_intel.columns]
    hw_keys = [k for k in group_keys if k in df_hw.columns]
    
    # Intelligence aggregation: mean for numeric, 'first' for model_response if it exists
    intel_agg_dict = {col: 'mean' for col in df_intel.select_dtypes(include=[np.number]).columns if col not in intel_keys}
    if 'model_response' in df_intel.columns:
        intel_agg_dict['model_response'] = 'first'
    
    df_intel_agg = df_intel.groupby(intel_keys).agg(intel_agg_dict).reset_index()
    df_hw_agg = df_hw.groupby(hw_keys).mean(numeric_only=True).reset_index()

    # Join on common keys
    common_keys = list(set(intel_keys) & set(hw_keys))
    df_combined = pd.merge(
        df_intel_agg, 
        df_hw_agg, 
        on=common_keys, 
        how='inner'
    )
    
    return df_combined

def export_to_csv(df, csv_path):
    """
    Exports the synthesized DataFrame to a CSV file.
    """
    if df is None or df.empty:
        print("[WARNING] DataFrame is empty. No CSV exported.")
        return
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    
    df.to_csv(csv_path, index=False)
    print(f"[SYSTEM] Metrics exported to {csv_path}")

def calculate_discontinuity(df, metric="passkey_retrieval_acc", method="elbow", threshold=0.95):
    """
    Identifies the lambda_b value where a performance discontinuity occurs.
    """
    if df is None or df.empty:
        return None
    
    # Sort by lambda_b
    df = df.sort_values("lambda_b").reset_index(drop=True)
    
    if method == "retention":
        # Find the last point that is >= threshold
        # We assume metric is higher is better (accuracy)
        # If perplexity (lower is better), we need a different logic
        if metric == "perplexity_score":
            # For perplexity, find the last point <= threshold * baseline
            baseline = df.iloc[0][metric]
            valid_points = df[df[metric] <= baseline / threshold]
        else:
            valid_points = df[df[metric] >= threshold]
            
        if valid_points.empty:
            return df.iloc[0]["lambda_b"]
        
        return valid_points.iloc[-1]["lambda_b"]

    elif method == "elbow":
        # Calculate second derivative to find the point of maximal curvature
        y = df[metric].values
        x = df["lambda_b"].values
        
        if len(x) < 3:
            return x[-1]
        
        # Calculate first and second derivatives
        dy = np.gradient(y, x)
        d2y = np.gradient(dy, x)
        
        # The discontinuity is where the second derivative is maximized (for drops)
        # or minimized (for spikes in PPL)
        idx = np.argmax(np.abs(d2y))
            
        return x[idx]
    
    return None

def generate_research_summary(df):
    """
    Generates a Markdown research summary from the synthesized DataFrame.
    """
    if df is None or df.empty:
        return "# Research Summary\n\nNo data available for analysis."
    
    summary = "# Research Summary\n\n"
    summary += "## Performance Metrics by Model and Lambda_B\n\n"
    
    # Select key columns for the summary table
    cols = ["model_name", "lambda_b", "passkey_retrieval_acc", "perplexity_score", "peak_vram_mb", "final_cache_size_mb"]
    # Filter to columns that actually exist in the df
    available_cols = [c for c in cols if c in df.columns]
    
    # Group by model and lambda_b to handle duplicates across multiple runs
    table_df = df[available_cols].groupby(["model_name", "lambda_b"]).mean().reset_index()
    table_df = table_df.sort_values(["model_name", "lambda_b"])
    
    summary += table_df.to_markdown(index=False)
    summary += "\n\n"
    
    summary += "## Discontinuity Analysis\n\n"
    for model in table_df["model_name"].unique():
        model_df = table_df[table_df["model_name"] == model]
        lb_acc = calculate_discontinuity(model_df, metric="passkey_retrieval_acc", method="elbow")
        lb_ppl = calculate_discontinuity(model_df, metric="perplexity_score", method="elbow")
        
        summary += f"### Model: {model}\n"
        summary += f"- **Discontinuity Threshold (Passkey):** λB = {lb_acc}\n"
        summary += f"- **Discontinuity Threshold (Perplexity):** λB = {lb_ppl}\n\n"
        
    return summary
