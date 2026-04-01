# PROJECT.md: The Lambda-B ($\lambda_B$) Evaluation Suite

## 1. Executive Summary & Objective
[cite_start]This project establishes a localized, machine-speed AI laboratory designed to empirically validate the **Bryan Coefficient ($\lambda_B$)**[cite: 263]. [cite_start]The objective is to transition from standard heuristic Key-Value (KV) cache eviction to a mathematically optimal time-decay protocol[cite: 264, 401]. [cite_start]By mapping the "Discontinuity Threshold"—the exact mathematical point where garbage collection of dead weights is maximized just before foundational attention sinks collapse [cite: 473, 477]—this suite will prove the absolute physical limit of context compression.

## 2. Hardware Constraints & Environment
[cite_start]The laboratory operates on consumer-grade hardware constrained by strict memory limits, proving the architecture's viability at the edge[cite: 300, 302].
* **Environment:** WSL2 (Ubuntu) strictly bounded via `.wslconfig` (12GB RAM limit, `swap=0`) to prevent host OS crashes during tensor calculations.
* [cite_start]**Precision:** Models must utilize 4-bit Quantization (via BitsAndBytes) to compress static weights and maximize available VRAM for the dynamic KV cache[cite: 313, 314].
* **Execution State:** All models must be initialized with `attn_implementation="eager"`. This is an absolute mandate. [cite_start]It disables fused FlashAttention kernels and exposes the raw Python mathematical graph for $\lambda_B$ injection[cite: 316, 318, 320].

## 3. The Target Model Pipeline
* [cite_start]**Prototyping Engine:** `Qwen/Qwen2-1.5B-Instruct` (Fast execution for rapid $\lambda_B$ iteration)[cite: 307].
* [cite_start]**Orchestration/Routing:** `microsoft/Phi-3-mini-4k-instruct`[cite: 307].
* **Publication Engine:** `unsloth/Meta-Llama-3.1-8B-bnb-4bit` (Must be loaded via standard Hugging Face `transformers` to avoid un-interceptable Triton kernels).

---

## 4. Operational Tracks

### Track 1: Pipeline Orchestration & Telemetry (The Scaffolding)
**Goal:** Establish the autonomous testing loop and persistence layer.
* **Components:** `run_experiments.sh` (Master bash script) and `init_db.py` (SQLite telemetry database).
* **Mechanics:** The bash script synchronously iterates through models and $\lambda_B$ gradients.
* **Survival Protocol:** Implements aggressive state-clearing (`cuda.empty_cache()`, `drop_caches`, and strict process killing) between every epoch.

### Track 2: Eager Mode Surgery (The Intercept)
[cite_start]**Goal:** Physically alter the model's fundamental attention mechanics via PyTorch monkey-patching[cite: 343].
* [cite_start]**The Injection:** Traverse to `LlamaAttention.forward()`[cite: 345]. [cite_start]Calculate the temporal distance ($\Delta t$) between current query and past keys[cite: 349]. 
* [cite_start]**The Math:** Apply the continuous decay penalty matrix directly to the `attn_weights` immediately prior to softmax normalization[cite: 264, 347, 356]. 
* [cite_start]**Garbage Collection:** Sum attention scores post-softmax to identify "Low-Value Vertices" and physically slice the Key and Value tensors to evict dead weights that fall below the threshold[cite: 366, 368, 369].

### Track 3: Discontinuity Mapping (The Micro-Sweep)
**Goal:** Isolate the exact "Amnesia Wall" where model logic collapses.
* **Strategy:** Abandon continuous curve fitting for **Micro-Sweep High-Resolution Bisection**. 
* **Zone Isolation:** Target the known fracture point in Llama-3.1-8B ($ \lambda_B \in [0.0010, 0.0013] $) and the Qwen2-1.5B "Attention Denoiser" peak ($ \lambda_B = 0.0001 $).
* **Identification:** Map the sudden phase transition where Passkey Accuracy collapses from 100% to 0% across a 0.00005 step.

### Track 4: The Hyperagent Evolutionary Loop (Metacognition)
**Goal:** Automate the tuning of the laboratory without triggering context bloat.
* **Components:** `GEMINI.md` (System DNA) and `mutate-dna.py` (The Immune System).
* **Mechanics:** Operates via a hardcoded state-swapping protocol. The execution phase dumps `epoch_results.txt`. The Orchestrator unloads, and the Hyperagent wakes up to analyze failures and rewrite the caching thresholds in `GEMINI.md`. 

## 5. Major Findings & Breakthroughs

### The Attention Denoiser Effect (2026-03-28)
An unexpected emergent property was discovered where applying a small Bryan Coefficient ($ \lambda_B = 0.0001 $) significantly **improves** accuracy for Small Language Models (SLMs) like Qwen2-1.5B in mid-length contexts. 
* **Conclusion:** $ \lambda_B $ acts as a signal-to-noise filter, physically removing low-importance attention tokens that create interference, thus mitigating "Lost in the Middle" errors.

### The "Amnesia Wall" Discontinuity
Telemetry reveals that model intelligence does not degrade gracefully. Instead, it hits a "cliff" (The Amnesia Wall) where the KV cache effectively collapses at a specific mathematical threshold.
* **Llama-3.1-8B Wall:** Identified between $ 0.0010 $ and $ 0.0013 $.
* **Resolution Requirement:** Finding the exact point of fracture to within $ \pm 0.00001 $ resolution to establish the "Maximum Safe Limit" for context compression.
