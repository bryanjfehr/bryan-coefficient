# Basic Agentic Evaluation (v1-v3)

This folder contains the scripts and documentation for the initial agentic evaluation phases where we discovered the limitations of Small Language Models (SLMs) in complex reasoning tasks, even without KV-cache pruning.

## Hardware Tiers
- **Google Colab (L4 GPU)**: Standard tier for executing the LangGraph SQL state machine.

## Methodology
These evaluations use a ReAct (Reasoning + Acting) framework implemented via LangGraph. The agent is tasked with interacting with a mock SQL environment to solve multi-step data retrieval and analysis problems.

### Key Discoveries
- **Amnesia Loops**: SLMs (1.5B - 3B) often fail to track state in long JSON payloads.
- **Baseline Performance**: Models exhibit reasoning failure even at $\lambda_B = 0$, indicating that KV-cache pruning is not the primary cause of logical collapse in early v1-v3 benchmarks.

## Reproduction Steps in Google Colab
1. Open a new Google Colab notebook with an **L4 GPU** runtime.
2. Run the provided `repro_basic_agent.py` script.
3. The script will:
   - Install `langgraph`, `langchain-core`, `outlines`, and other dependencies.
   - Set up a mock SQL database for the agent.
   - Execute a series of reasoning turns and log the successes/failures.

## Files
- `repro_basic_agent.py`: Self-contained script for agentic evaluation reproduction.
- `metadata.json`: Technical parameters for this evaluation tier.
