# Bryan Coefficient Evaluation Suite (Reproduction)

This directory contains the official reproduction suite for the Bryan Coefficient ($\lambda_B$) research. These scripts are designed to replicate the findings in the `results/` folder with academic rigor.

## Master Orchestration Scripts
Researchers should use the following master scripts located in the project root:

1. **`bash run_experiments.sh`**: Reproduces the scalar bisection, entropy extraction, and multidimensional benchmarks.
2. **`bash run_agent_test.sh`**: Reproduces the agentic reasoning benchmarks (SQL retrieval) under KV-cache decay.

## Evaluation Tiers & Hardware Requirements

| Tier | Focus | Recommended Hardware | Models |
| :--- | :--- | :--- | :--- |
| **[Small Needle-in-a-Haystack](./eval_small_needle_haystack/)** | Discontinuity Mapping | RTX 4060 / L4 GPU | Qwen2.5-1.5B, Phi-3-mini |
| **[Basic Agentic (v1-v3)](./eval_basic_agent_v1_v3/)** | Reasoning Failure (Amnesia) | L4 GPU (16GB VRAM) | Gemma-2-2B, Llama-3.2-3B |
| **[Large Model Agentic](./eval_large_agent_a100/)** | High-Res Bisection & Compression | A100 GPU (40GB/80GB) | Llama-3.1-8B, Mistral-7B |

## Core Methodology
- **Eager Execution**: All evaluations MUST use `attn_implementation="eager"` to enable physical KV-cache surgery.
- **4-bit Quantization**: BitsAndBytes (NF4) is used to fit models into consumer/standard cloud VRAM.
- **Physical Eviction**: Unlike "soft" masking, these scripts perform actual VRAM recovery by slicing the KV-cache tensors.

## Reproduction Protocol
To reproduce the findings from scratch:

1. **Environment Setup**: Ensure all dependencies are installed (see `requirements.txt`).
2. **Authentication**: Set `HF_TOKEN` as an environment variable or in Colab Secrets.
3. **Execution**:
   ```bash
   # Phase 1: Map the Discontinuity Thresholds
   bash run_experiments.sh
   
   # Phase 2: Verify Reasoning Integrity
   bash run_agent_test.sh
   ```
4. **Validation**: Compare the generated `metrics.csv` and `results/agent_results_*/` with the baseline findings in the `results/` folder.
