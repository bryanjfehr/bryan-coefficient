# Large Model Agentic Evaluation (A100)

This evaluation tier reproduces the $\lambda_B$ (Bryan Coefficient) protocol for large language models (8B - 70B+). It is optimized for high-compute environments like Google Colab with an NVIDIA A100 GPU.

## Hardware Requirements
- **Google Colab (A100 GPU)**: Mandatory tier for models like Llama-3.1-8B and above to avoid OOM and ensure reasonable inference speeds with 4-bit quantization.

## Methodology
The reproduction process follows a two-phase protocol:
1.  **Phase 1 (Bisection Search)**: Automatically identify the "Discontinuity Cliff" ($\lambda_B$ boundary) where Passkey Retrieval accuracy begins to degrade.
2.  **Phase 2 (Agentic Stress Test)**: Execute a multi-step SQL reasoning task at the identified boundary to verify that reasoning integrity holds even near the cliff.

## Reproduction Steps

1.  Open a new Google Colab notebook with an **A100 GPU** runtime.
2.  Ensure you have the project folder in your Google Drive at `/content/drive/MyDrive/bryan-coefficient`.
3.  Authenticate using your `HF_TOKEN`.
4.  Run the provided `repro_large_agent.sh` script:
    ```bash
    !bash repro_evals/eval_large_agent_a100/repro_large_agent.sh
    ```

## Expected Results
- **Discontinuity Cliff**: Large models like Llama-3.1-8B exhibit a much higher tolerance for $\lambda_B$ (often up to 0.005 - 0.01) compared to SLMs.
- **Reasoning Preservation**: Even with significant KV-cache eviction (often 50%+), these models should maintain high success rates in the SQL agentic task.

## Key Files
- `repro_large_agent.sh`: Master orchestration script for high-compute evaluation.
- `bisection_search.py`: Core utility for finding the discontinuity threshold.
- `run_benchmarks.py`: The primary evaluation engine used for both Passkey and Agentic tests.
