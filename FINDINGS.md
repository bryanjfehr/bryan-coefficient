# EMPIRICAL FINDINGS: PROJECT BRYAN COEFFICIENT

## 1. The Attention Denoiser Effect (Discovery: 2026-03-28)
**Observation:** Qwen2-1.5B-Instruct fails the Passkey Retrieval benchmark (4,000 characters) at baseline ($ \lambda_B = 0 $) but achieves 100% accuracy at $ \lambda_B = 0.0001 $.

**Mechanism:** 
- At $ \lambda_B = 0.0001 $, the surgery prunes between 70% and 90% of the KV cache across various layers.
- Despite this massive loss of "memory," the model's retrieval accuracy increases from 0% to 100%.

**Hypothesis:** 
The Bryan Coefficient acts as a high-pass filter for attention. Small Language Models (SLMs) like Qwen2-1.5B suffer from significant "attention noise" in long contexts, causing them to lose signal (the passkey) even when the input is well within their native 32k context window. By aggressively pruning low-value tokens, $ \lambda_B $ denoises the query path, mitigating the "Lost in the Middle" phenomenon.

## 2. Model-Specific Discontinuity Thresholds
*(Data pending full Gauntlet run)*

## 3. Agentic Framework Assessment (Discovery: 2026-03-29)
**Observation:** The current agentic evaluation framework exhibits a "Reasoning-to-Capability Disconnect" for Small Language Models (SLMs) in the 1.5B–3B range, even at zero KV decay ($\lambda_B = 0$).

**Key Failure Modes:**
- **Instruction Drift (Qwen2.5-1.5B):** Failed to recognize the multi-step nature of the task, attempting to guess the passcode in Turn 1 without tool use.
- **Schema Fragility (Phi-3-mini):** Encountered `EOF` parsing errors despite Outlines enforcement, suggesting truncation or failure to close JSON structures under high token pressure.
- **"Large Observation" Wall (Gemma-2-2B, Llama-3.1-8B):** Large JSON payloads from tools (e.g., employee database dumps) caused models to lose task context, leading to "Amnesia Loops" (repetitive tool calls).
- **Alignment-Induced Refusal (Llama-3.2-3B):** The safety-tuned model identified the "security auditor" roleplay as a potential intrusion attempt and refused the baseline task.
- **Synthesis Collapse (Mistral-7B):** Successful navigation through 4 turns of tool use followed by a failure at the final step (malformed file decryption attempt).

**Conclusion:** The current ReAct gauntlet is tuned beyond the zero-decay capability floor of sub-7B models. Observation noise (large tool outputs) and strict JSON enforcement cause logical collapse independent of KV cache pressure.

**Recommendation for Epoch 1:**
1. **Observation Summarization:** Truncate or summarize tool outputs to reduce context saturation.
2. **Few-Shot Stabilization:** Transition to few-shot ReAct prompts to anchor the reasoning trajectory.
3. **Persona Pivot:** Replace the "Auditor" persona with a neutral "Network Administrator" role to mitigate safety-triggered refusals in Llama-3.2.

## 4. Epoch 1 Post-Mortem & The Amnesia Loop (Discovery: 2026-03-30)
**Observation:** Following the implementation of Epoch 1 fixes (Few-Shot Prompting, Outlines v1 API, Observation Summarization, Neutral Persona), syntax and alignment issues were resolved. Models successfully formatted JSON tool calls and bypassed safety refusals. However, all evaluated SLMs (Qwen2.5-1.5B, Phi-3-mini, Gemma-2-2B, Llama-3.1-8B, Llama-3.2-3B, Mistral-7B) uniformly failed due to **"Amnesia Loops."**

**Mechanism of Failure:**
- Models successfully executed the initial `get_employee_database` tool call.
- Upon receiving the truncated observation (or even standard observations), models failed to extract the correct ID (e.g., EMP8842) or maintain the multi-step intent.
- Instead of progressing to `get_server_logs`, models repeated the exact same tool call, queried invalid tool names, or hallucinated arguments.
- **Root Cause:** The models lack intrinsic persistent state management and robust Chain-of-Thought (CoT) grounding. The immediate jump to outputting a JSON tool call prevents the model from "thinking out loud" to update its internal state.

**Conclusion:** Relying purely on the LLM's autoregressive sequence to maintain ReAct state is fundamentally flawed for sub-7B models under evaluation. 

**Recommendation for Epoch 2 (AGENTEVAL_v3):**
Transition the evaluation framework to a **LangGraph-based state machine**. 
- Implement explicit persistent state.
- Decouple "Reasoning" from "Tool Calling" to enforce CoT prior to JSON generation.
- Use LangGraph to manage the conversational memory structure, ensuring that the Bryan Coefficient's impact is measured against actual retrieval/synthesis degradation, not syntax or state-tracking fragility.
