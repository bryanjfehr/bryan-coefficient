import os
import argparse
from datetime import datetime
from analysis_utils import fetch_metrics, export_to_csv, generate_research_summary
from visualization_utils import plot_pareto_frontier, plot_model_comparisons, plot_evolutionary_trends

def main():
    parser = argparse.ArgumentParser(description="Bryan Coefficient Discontinuity Mapping Pipeline")
    parser.add_argument("--db", type=str, default="lambda_b_telemetry.db", help="Path to telemetry database")
    parser.add_argument("--output-dir", type=str, default=None, help="Base directory for results")
    args = parser.parse_args()

    # Create dated output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        results_dir = os.path.join(args.output_dir, f"results_{timestamp}")
    else:
        results_dir = os.path.join("results", f"results_{timestamp}")
    
    os.makedirs(results_dir, exist_ok=True)
    print(f"[SYSTEM] Initializing mapping pipeline. Output: {results_dir}")

    # 1. Fetch data
    try:
        df = fetch_metrics(args.db)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return

    if df.empty:
        print("[WARNING] No telemetry data found in database. Exiting.")
        return

    print(f"[SYSTEM] Fetched {len(df)} telemetry records.")

    # 2. Export CSV
    csv_path = os.path.join(results_dir, "metrics.csv")
    export_to_csv(df, csv_path)

    # 3. Generate Visualizations
    print("[SYSTEM] Generating visualizations...")
    plot_pareto_frontier(df, os.path.join(results_dir, "pareto_frontier.png"))
    plot_model_comparisons(df, os.path.join(results_dir, "model_comparisons.png"))
    plot_evolutionary_trends(df, os.path.join(results_dir, "evolutionary_trends.png"))

    # 4. Generate Research Summary
    print("[SYSTEM] Synthesizing research summary...")
    summary = generate_research_summary(df)
    summary_path = os.path.join(results_dir, "research_summary.md")
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"[SYSTEM] Research summary saved to {summary_path}")

    # 5. Latest symlink
    latest_link = "results/latest"
    if os.path.lexists(latest_link):
        if os.path.islink(latest_link):
            os.unlink(latest_link)
        else:
            # If it's a real dir, don't delete it
            pass
    
    try:
        os.symlink(os.path.abspath(results_dir), latest_link)
        print(f"[SYSTEM] Created symlink: {latest_link} -> {results_dir}")
    except Exception as e:
        print(f"[WARNING] Could not create symlink: {e}")

    print("[SUCCESS] Discontinuity Mapping complete.")

if __name__ == "__main__":
    main()
