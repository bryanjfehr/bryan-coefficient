import pandas as pd
import os

metrics_file = 'results/results_20260329_075158/metrics.csv'
if not os.path.exists(metrics_file):
    print(f"Error: {metrics_file} not found")
    exit(1)

df = pd.read_csv(metrics_file)
# Normalize model names if necessary (sometimes they have leading/trailing spaces or ./ )
df['model_name'] = df['model_name'].str.strip().str.replace('./', '', regex=False)

models = df['model_name'].unique()
results = {}

for m in models:
    m_df = df[df['model_name'] == m]
    # We want the highest lambda_b that maintains intelligence
    # Intelligence = 100% passkey retrieval AND Perplexity degradation < 2x (baseline is approx 1.2)
    # Actually let's just look for Passkey = 1.0 first as it's the stricter amnesia wall.
    success = m_df[(m_df['passkey_retrieval_acc'] == 1.0)]
    
    if not success.empty:
        max_lb = success['lambda_b'].max()
        results[m] = max_lb
    else:
        # If no 1.0 success, find the absolute best they did
        best_acc = m_df['passkey_retrieval_acc'].max()
        results[m] = f"FAILED (Best Acc: {best_acc})"

print("--- DISCONTINUITY THRESHOLD ANALYSIS ---")
for m, lb in results.items():
    print(f"{m}: {lb}")
