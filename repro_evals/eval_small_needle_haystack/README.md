# Small Model Needle-in-a-Haystack Evaluation

This evaluation tier reproduces the $\lambda_B$ (Bryan Coefficient) discontinuity mapping for small language models (SLMs). It identifies the specific threshold where KV-cache eviction leads to retrieval failure.

## Hardware Requirements
- **Local (RTX 4060)** or **Google Colab (L4 GPU)**: Standard tier for mapping discontinuity thresholds across 8k-12k context windows.

## Reproduction Steps

### Option A: Manual Targeted Sweep
To map the discontinuity cliff for a specific model (e.g., Qwen2.5-1.5B):

1.  Open a Google Colab notebook with an **L4 GPU**.
2.  Mount Google Drive and authenticate with Hugging Face.
3.  Execute a targeted evaluation:
    ```bash
    # Test λB = 0.001 (Should Pass)
    python repro_evals/eval_small_needle_haystack/repro_small_needle.py --model Qwen/Qwen2.5-1.5B-Instruct --lambda_b 0.001
    
    # Test λB = 0.002 (Should Fail/Degrade)
    python repro_evals/eval_small_needle_haystack/repro_small_needle.py --model Qwen/Qwen2.5-1.5B-Instruct --lambda_b 0.002
    ```

### Option B: Full Evolutionary Gauntlet
To reproduce the complete scalar and multidimensional results (including bisection search) across all models:

1.  Run the master orchestration script:
    ```bash
    bash run_experiments.sh
    ```

## Expected Results
- **Scalar Cliff**: You should observe a sharp drop in Passkey Retrieval accuracy as $\lambda_B$ crosses the model's specific threshold (typically between 0.001 and 0.005 for SLMs).
- **Perplexity Stability**: WikiText-103 Perplexity (PPL) should remain relatively stable until just before the cliff, where it will spike exponentially, indicating catastrophic linguistic collapse.
- **VRAM Recovery**: The logs will report "Final KV Cache Size" in MB, showing the actual memory saved compared to a baseline run (`--lambda_b 0.0`).

## Key Files
- `repro_small_needle.py`: Orchestrates a real needle-in-a-haystack run using the project's core `run_benchmarks.py` logic.
- `bisection_search.py`: Automated tool used by `run_experiments.sh` to find the exact cliff point.
