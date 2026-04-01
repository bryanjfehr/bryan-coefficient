# Basic Agentic Evaluation (v1-v3)

This evaluation tier reproduces the discovery of **Amnesia Loops** in Small Language Models (SLMs). It demonstrates that SLMs often fail at multi-step reasoning even without KV-cache pruning ($\lambda_B = 0$).

## Hardware Requirements
- **L4 GPU** (16GB VRAM): Minimum requirement for running 2B-3B models in 4-bit eager mode.

## Reproduction Steps

### Option A: Manual Targeted Run
To reproduce the finding on a specific model (e.g., Gemma-2-2B):

1.  Open a Google Colab notebook with an **L4 GPU**.
2.  Mount Google Drive and authenticate with Hugging Face.
3.  Run the following:
    ```bash
    python repro_evals/eval_basic_agent_v1_v3/repro_basic_agent.py --model google/gemma-2-2b-it --lambda_b 0.0
    ```

### Option B: Full Agentic Gauntlet
To reproduce the complete agentic findings across all models as seen in the `results/` folder:

1.  Run the master orchestration script:
    ```bash
    bash run_agent_test.sh
    ```

## Expected Results
- **Amnesia Loops**: Models like Gemma-2-2B or Qwen2.5-1.5B will frequently fail to progress beyond the first SQL tool call, repeating the same query or losing track of the JSON schema.
- **Baseline Integrity**: This evaluation confirms that the "failure cliff" observed in larger models is a distinct phenomenon from the baseline reasoning limitations of SLMs.

## Key Files
- `repro_basic_agent.py`: Orchestrates a real agentic run using the project's core `run_benchmarks.py` logic.
- `sql_eval_setup.py`: Generates the live SQLite environment used for the evaluation.
