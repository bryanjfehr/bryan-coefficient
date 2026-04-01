# Large Model Agentic Evaluation (A100)

This folder contains the scripts and documentation for evaluating large language models (20B - 30B+) under the λB (Bryan Coefficient) protocol. This tier is optimized for high-compute environments like Google Colab with an NVIDIA A100 GPU.

## Hardware Tiers
- **Google Colab (A100 GPU)**: Mandatory tier for large models to avoid OOM and ensure reasonable inference speeds with 4-bit quantization.

## Methodology
Large models are evaluated using the V4 Progressive Gauntlet protocol, which includes automated bisection search to find the discontinuity cliff and a progressive agentic sweep using LangGraph.

### Key Performance Indicators (KPIs)
- **Discontinuity Cliff**: The λB value where Passkey accuracy drops sharply.
- **VRAM Recovery**: The amount of KV-cache memory reclaimed via physical eviction.
- **Agent Success**: The ability of the model to maintain tool-use accuracy near the cliff.

## Reproduction Steps in Google Colab
1. Open a new Google Colab notebook with an **A100 GPU** runtime.
2. Ensure you have the project folder in your Google Drive at `/content/drive/MyDrive/bryan-coefficient`.
3. Ensure your `HF_TOKEN` is saved in Colab Secrets and that you have model access permissions.
4. Run the following bootstrap script in a cell to mount your drive and authenticate:

```python
from google.colab import drive, userdata
import os
import sys

# 1. Mount Drive
drive.mount('/content/drive')
project_path = '/content/drive/MyDrive/bryan-coefficient'
if os.path.exists(project_path):
    os.chdir(project_path)
    sys.path.append(project_path)
    print(f"Project path: {os.getcwd()}")
else:
    print(f"Error: {project_path} not found.")

# 2. HF Login
from huggingface_hub import login
hf_token = userdata.get('HF_TOKEN')
login(token=hf_token)
```

5. Run the provided `repro_large_agent.sh` script in a cell:
   ```bash
   !bash repro_evals/eval_large_agent_a100/repro_large_agent.sh
   ```

## Files
- `repro_large_agent.sh`: Master orchestration script for large model evaluation.
- `metadata.json`: Technical parameters for this evaluation tier.
