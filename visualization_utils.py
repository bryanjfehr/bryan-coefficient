import plotly.express as px
import plotly.graph_objects as go
import os
import pandas as pd

def plot_pareto_frontier(df, output_path="results/pareto.png"):
    """
    Plots Cache size against Passkey Accuracy to identify optimal operating points.
    """
    if df is None or df.empty:
        return None
    
    # Use final_cache_size_mb if available, otherwise fallback to peak_vram_mb
    x_col = "final_cache_size_mb" if "final_cache_size_mb" in df.columns else "peak_vram_mb"
    x_label = "Final KV Cache Size (MB)" if x_col == "final_cache_size_mb" else "Peak VRAM (MB)"

    # Create scatter plot
    fig = px.scatter(
        df, 
        x=x_col, 
        y="passkey_retrieval_acc", 
        color="model_name",
        hover_data=["lambda_b", "peak_vram_mb"],
        title=f"Pareto Frontier: {x_label} vs Intelligence (Passkey Accuracy)",
        labels={x_col: x_label, "passkey_retrieval_acc": "Passkey Accuracy"}
    )
    
    # Add trendlines or specific formatting if needed
    fig.update_layout(template="plotly_white")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Export to static image
    fig.write_image(output_path, scale=2)
    print(f"[SYSTEM] Pareto plot exported to {output_path}")
    
    return fig

def plot_model_comparisons(df, output_path="results/comparisons.png"):
    """
    Visualizes hardware and intelligence deltas across models.
    """
    if df is None or df.empty:
        return None
    
    # Create a grouped bar chart for a specific lambda_b or average
    # For now, let's plot Cache Size at the baseline lambda_b (usually minimum)
    baseline_df = df.sort_values("lambda_b").groupby("model_name").first().reset_index()
    
    y_col = "final_cache_size_mb" if "final_cache_size_mb" in df.columns else "peak_vram_mb"
    y_label = "Baseline KV Cache Size (MB)" if y_col == "final_cache_size_mb" else "Peak VRAM (MB)"

    fig = px.bar(
        baseline_df,
        x="model_name",
        y=y_col,
        title=f"Model {y_label} Baseline",
        labels={"model_name": "Model", y_col: y_label},
        color="model_name"
    )
    
    fig.update_layout(template="plotly_white")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Export to static image
    fig.write_image(output_path, scale=2)
    print(f"[SYSTEM] Comparisons plot exported to {output_path}")
    
    return fig

def plot_evolutionary_trends(df, output_path="results/evolution.png"):
    """
    Tracks the evolution of lambda_b performance across multiple experimental epochs.
    """
    if df is None or df.empty or "epoch" not in df.columns:
        return None
    
    # Aggregate by epoch and model
    # For evolution, we might want to look at the maximum VRAM saving achieved
    # or the average accuracy at a fixed lambda_b.
    # For simplicity, we plot the mean accuracy per epoch
    epoch_df = df.groupby(["epoch", "model_name"]).mean(numeric_only=True).reset_index()
    
    fig = px.line(
        epoch_df,
        x="epoch",
        y="passkey_retrieval_acc",
        color="model_name",
        title="Evolutionary Intelligence Trend (Mean Passkey Accuracy)",
        labels={"epoch": "Epoch", "passkey_retrieval_acc": "Mean Passkey Accuracy"},
        markers=True
    )
    
    fig.update_layout(template="plotly_white")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Export to static image
    fig.write_image(output_path, scale=2)
    print(f"[SYSTEM] Evolutionary trend plot exported to {output_path}")
    
    return fig
