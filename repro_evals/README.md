# Bryan Coefficient Evaluation Suite (Reproduction)

This directory contains a tiered evaluation suite for reproducing the findings of the Bryan Coefficient ($\lambda_B$) research. Each tier is optimized for specific hardware constraints and evaluation goals.

## Evaluation Tiers
1. **[Small Model Needle-in-a-Haystack](./eval_small_needle_haystack/)**: 
   - **Focus**: Mapping the initial discontinuity threshold.
   - **Hardware**: RTX 4060, L4 GPU.
   - **Metrics**: Passkey Accuracy, Perplexity.
2. **[Basic Agentic Evaluation (v1-v3)](./eval_basic_agent_v1_v3/)**: 
   - **Focus**: Identifying reasoning failure modes (Amnesia Loops).
   - **Hardware**: L4 GPU.
   - **Metrics**: SQL retrieval success, reasoning turns.
3. **[Large Model Agentic Evaluation (A100)](./eval_large_agent_a100/)**: 
   - **Focus**: High-resolution bisection and context compression for 20B+ models.
   - **Hardware**: A100 GPU.
   - **Metrics**: VRAM recovery, agent success near the cliff.

## Core Mandates
- **Eager Mode**: All evaluations MUST use `attn_implementation="eager"` to allow KV-cache surgery.
- **4-bit Quantization**: BitsAndBytes (NF4) is the default for fitting into 8GB - 16GB VRAM limits.
- **Sink Protection**: Preservation of foundational tokens is mandatory for model stability.

## Running in Google Colab
Each tier provides scripts optimized for Google Colab. To ensure success:

1. **Google Drive Integration**: The scripts expect the project to be located at `/content/drive/MyDrive/bryan-coefficient`. They will automatically attempt to mount your drive and add this path to `sys.path`.
2. **Hugging Face Authentication**: You must have `HF_TOKEN` saved in your Google Colab **Secrets** (the key icon in the sidebar). Ensure the "Notebook access" toggle is turned on for this secret.
3. **Model Permissions**: Ensure your Hugging Face account has been granted access to gated models from Meta (Llama), Google (Gemma), and Microsoft (Phi).
4. **Hardware**: Select the appropriate GPU runtime (L4 or A100) as specified in each tier's README.
