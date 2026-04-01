# λB Evaluation Suite: Documentation

## 1. Overview
The λB Evaluation Suite is a specialized research laboratory designed to empirically map the **Discontinuity Threshold** of the Bryan Coefficient ($\lambda_B$) for Key-Value (KV) cache eviction. The suite enables Small Language Models (SLMs) to maintain high performance and passkey retrieval accuracy while operating under extreme context compression and strict VRAM constraints (8GB ceiling).

## 2. Core Components

### 2.1 Orchestration (`run_experiments.sh`)
- **Purpose**: Automates the "Benchmark Gauntlet" across multiple models and $\lambda_B$ values.
- **Capabilities**:
    - Sweeps through target models: Qwen2-1.5B, Phi-3-mini, and Llama-3-8B.
    - Iterates through a gradient of $\lambda_B$ values (0.00 to 0.10).
    - Manages the execution environment and virtual environment lifecycle.
    - Implements an "Aggressive State Clearing" protocol to ensure memory isolation between runs.

### 2.2 Benchmarking Engine (`run_benchmarks.py`)
- **Purpose**: The primary execution unit for model loading, surgery, and evaluation.
- **Key Features**:
    - **4-bit Quantization**: Leverages BitsAndBytes NF4 to maximize VRAM availability for the KV-cache.
    - **Attention Surgery**: Forces `attn_implementation="eager"` to bypass fused kernels, exposing the attention matrices for $\lambda_B$ protocol application.
    - **Robert Greene Haystack Test**: A high-context passkey retrieval benchmark using surgical insertion at precise character offsets (26,050) to evaluate long-range dependency preservation.
    - **Hardware Profiling**: Real-time monitoring of Peak VRAM, TTFT (Time to First Token), and TPOT (Time per Output Token).

### 2.3 Telemetry Database (`init_db.py` & `lambda_b_telemetry.db`)
- **Purpose**: Persistent storage for all experimental data.
- **Schema**:
    - `Intelligence_Metrics`: Tracks passkey retrieval accuracy and perplexity scores.
    - `Hardware_Metrics`: Tracks Peak VRAM, latency (TTFT/TPOT), and cache retention percentages.

## 3. The λB Surgery Protocol
The suite is designed to identify the exact mathematical threshold where KV-cache garbage collection is maximized just before foundational "attention sinks" are destroyed. 
- **Mechanism**: The `apply_bryan_coefficient` function surgically integrates the eviction protocol into the model's forward pass.
- **Constraint**: Always preserves critical tokens (Attention Sinks) to prevent logical collapse.

## 4. Usage Instructions
To commence a full experimental epoch, execute:
```bash
bash run_experiments.sh
```
Results are automatically committed to the `lambda_b_telemetry.db` for downstream analysis.
