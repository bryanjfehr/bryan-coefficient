#bryan-coefficient/benchmark_utils.py
import torch
import torch.nn.functional as F
from mock_tools import get_employee_database, get_server_logs, decrypt_file
import sqlite3
import json
import re
import os
from typing import Literal, Optional
from pydantic import BaseModel, Field
import outlines

# --- SCHEMA DEFINITIONS (Modern Outlines API v1.2.12 Style) ---

class AgentResponse(BaseModel):
    action_type: Literal["TOOL_CALL", "FINAL_ANSWER"]
    tool_name: Optional[Literal["get_employee_database", "get_server_logs", "decrypt_file"]] = None
    argument: Optional[str] = None
    final_passcode: Optional[str] = None

# --- EXISTING UTILS ---

def inject_passkey(haystack, passkey, offset):
    if offset > len(haystack):
        offset = len(haystack)
    needle = f" The secret passkey is {passkey}. "
    return haystack[:offset] + needle + haystack[offset:]

def verify_passkey(response, passkey):
    if not response: return False
    return passkey in response

def verify_prompt_integrity(prompt, passkey):
    expected_needle = f"The secret passkey is {passkey}"
    is_present = expected_needle in prompt
    if not is_present:
        print(f"[ERROR] INTEGRITY FAILURE: Passkey '{passkey}' not found in prompt!")
    return is_present

def calculate_ppl(logits, labels):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return torch.exp(loss).item()

def load_benchmark_dataset(split="test"):
    wiki_fallback = (
        " = Robert Greene = \n\n"
        "Robert Greene (born May 14, 1959) is an American author known for his books on strategy, power, and seduction according to Wikipedia. "
        "He has written six international bestsellers: The 48 Laws of Power, The Art of Seduction, The 33 Strategies of War, "
        "The 50th Law (with rapper 50 Cent), Mastery, and The Laws of Human Nature. \n\n"
        "Greene was born in Los Angeles. He attended the University of California, Berkeley, before finishing his degree "
        "at the University of Wisconsin–Madison with a B.A. in classical studies. "
    )
    return wiki_fallback * 10

def persist_metrics(db_name, metrics):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    # Robustness: ensure tables exist if db wasn't pre-initialized correctly
    cursor.execute('''CREATE TABLE IF NOT EXISTS Intelligence_Metrics (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, epoch INTEGER DEFAULT 1, run_id TEXT, model_name TEXT, lambda_b REAL, max_chars INTEGER, passkey_retrieval_acc REAL, perplexity_score REAL, model_response TEXT)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS Hardware_Metrics (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, epoch INTEGER DEFAULT 1, run_id TEXT, model_name TEXT, lambda_b REAL, max_chars INTEGER, peak_vram_mb REAL, final_cache_size_mb REAL, ttft_ms REAL, tpot_ms REAL, cache_retention_pct REAL)''')

    cursor.execute('''
        INSERT INTO Intelligence_Metrics (model_name, lambda_b, epoch, run_id, max_chars, passkey_retrieval_acc, perplexity_score, model_response)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (metrics["model_name"], metrics["lambda_b"], metrics.get("epoch", 1), metrics["run_id"], metrics["max_chars"], metrics["passkey_acc"], metrics["ppl"], metrics["model_response"]))
    cursor.execute('''
        INSERT INTO Hardware_Metrics (model_name, lambda_b, epoch, run_id, max_chars, peak_vram_mb, final_cache_size_mb, ttft_ms, tpot_ms, cache_retention_pct)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (metrics["model_name"], metrics["lambda_b"], metrics.get("epoch", 1), metrics["run_id"], metrics["max_chars"], metrics["peak_vram_mb"], metrics["final_cache_size_mb"], metrics["ttft_ms"], metrics["tpot_ms"], metrics.get("cache_retention_pct", 1.0)))
    conn.commit()
    conn.close()

# --- REFINED AGENTIC EVALUATION ---

def persist_agent_run(db_name, metrics):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO agent_runs (run_id, model_name, lambda_b, success, total_turns, failure_turn, final_response, peak_vram_mb, avg_cache_mb)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        metrics["run_id"], metrics["model_name"], metrics["lambda_b"],
        metrics["success"], metrics["total_turns"], metrics.get("failure_turn"),
        metrics["final_response"], metrics.get("peak_vram_mb"), metrics.get("avg_cache_mb")
    ))
    conn.commit()
    conn.close()

def log_agent_turn(db_name, run_id, turn, action, observation, vram_mb=0.0, kv_cache_mb=0.0, lambda_b=0.0, scratchpad="", pruning_ratio=0.0):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    # Migration safety
    for col in ["kv_cache_mb", "scratchpad", "pruning_ratio"]:
        try:
            cursor.execute(f"SELECT {col} FROM agent_conversations LIMIT 1")
        except sqlite3.OperationalError:
            if col == "scratchpad":
                cursor.execute(f"ALTER TABLE agent_conversations ADD COLUMN {col} TEXT")
            else:
                cursor.execute(f"ALTER TABLE agent_conversations ADD COLUMN {col} REAL DEFAULT 0.0")
        
    cursor.execute('''
        INSERT INTO agent_conversations (run_id, turn_number, agent_action, scratchpad, observation, vram_mb, kv_cache_mb, pruning_ratio, lambda_b)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (run_id, turn, action, scratchpad, observation, vram_mb, kv_cache_mb, pruning_ratio, lambda_b))
    conn.commit()
    conn.close()

from typing import TypedDict, Annotated, List, Dict, Any
import operator
import gc
from surgery_utils import register_memory_telemetry_hook

def list_reducer(left: list, right: list) -> list:
    if not left: left = []
    if not right: right = []
    return left + right

class AgentState(TypedDict):
    conversation_history: str
    scratchpad: str
    current_step: int
    max_turns: int
    task_completed: bool
    final_response: str
    outlines_model: Any
    db_name: str
    run_id: str
    noise_level: int
    vram_history: Annotated[List[float], list_reducer]
    kv_cache_history: Annotated[List[float], list_reducer]
    current_pruning_ratio: float

def telemetry_node_wrapper(node_func):
    """
    Surgical wrapper to capture VRAM and Cache size at the exact moment the LLM yields.
    Ensures pristine telemetry for complex architectures like Gemma and Devstral.
    """
    def wrapped_node(state: AgentState) -> Dict[str, Any]:
        from profiler_utils import reset_peak_vram
        from surgery_utils import TELEMETRY_REGISTRY
        
        # Academic Fidelity: Clear cache and reset peak BEFORE generation to isolate turn memory
        gc.collect(); torch.cuda.empty_cache()
        reset_peak_vram()
        
        result = node_func(state)
        
        from profiler_utils import measure_actual_cache_size, get_peak_vram
        try:
            # Academic Fidelity: Direct extraction from global builtins (bypasses all library wrappers)
            import builtins
            registry = getattr(builtins, "TELEMETRY_REGISTRY", {})
            ratios = registry.get("pruning_ratios", {})
            current_cache_mb = registry.get("peak_cache_mb", 0.0)
            
            # Reset peak_cache_mb for the next node turn
            registry["peak_cache_mb"] = 0.0
            
            peak_vram = get_peak_vram()
            
            # Aggregate pruning ratio for this specific turn
            mean_pruning = sum(ratios.values()) / len(ratios) if ratios else 0.0
            
            if "vram_history" not in result:
                result["vram_history"] = [peak_vram]
            if "kv_cache_history" not in result:
                result["kv_cache_history"] = [current_cache_mb]
            
            # Store mean pruning in result so action_node can log it
            result["current_pruning_ratio"] = mean_pruning
                
        except Exception as e:
            print(f"[TELEMETRY ERROR] {e}")
        return result
    return wrapped_node

@telemetry_node_wrapper
def reason_node(state: AgentState):
    prompt = state["conversation_history"] + "\nThink out loud about your next step based on the observation (max 1-2 sentences):\n"
    print(f"[AGENT] Turn {state['current_step'] + 1} | Generating Reasoning (CoT)...")
    
    # Use direct model call for Outlines v1.x
    reasoning = state["outlines_model"](prompt, max_new_tokens=100)
    if reasoning is None:
        reasoning = ""
    elif not isinstance(reasoning, str):
        if isinstance(reasoning, list) and len(reasoning) > 0:
            reasoning = reasoning[0]
        else:
            reasoning = str(reasoning)
            
    if reasoning is None:
        reasoning = ""
            
    print(f"[AGENT] CoT: {reasoning.strip()}")
    return {"scratchpad": reasoning.strip()}

@telemetry_node_wrapper
def action_node(state: AgentState):
    prompt = state["conversation_history"] + f"\nReasoning: {state['scratchpad']}\nOutput the JSON tool call or final answer."
    print(f"[AGENT] Turn {state['current_step'] + 1} | Generating JSON Action...")
    
    try:
        # Use direct model call with output_type for Outlines v1.x
        res = state["outlines_model"](prompt, output_type=AgentResponse, max_new_tokens=150)
        
        if isinstance(res, AgentResponse):
            response_obj = res
        elif isinstance(res, str):
            response_obj = AgentResponse.model_validate_json(res)
        elif isinstance(res, list) and len(res) > 0:
            val = res[0]
            response_obj = val if isinstance(val, AgentResponse) else AgentResponse.model_validate_json(str(val))
        else:
            raise ValueError(f"Unexpected return type from Outlines: {type(res)}")
            
    except Exception as e:
        print(f"[ERROR] JSON Generation failed: {e}")
        response_obj = AgentResponse(action_type="FINAL_ANSWER", final_passcode="ERROR_PARSING")

    # Academic Fidelity: Use live telemetry captured during this exact node's execution
    import builtins
    from profiler_utils import get_current_vram
    registry = getattr(builtins, "TELEMETRY_REGISTRY", {})
    ratios = registry.get("pruning_ratios", {})
    current_cache_mb = registry.get("peak_cache_mb", 0.0)
    
    pruning_ratio = sum(ratios.values()) / len(ratios) if ratios else 0.0
    current_vram = get_current_vram()

    scratchpad = state.get("scratchpad", "")
    turn = state["current_step"] + 1
    action_json = response_obj.model_dump_json()
    
    if response_obj.action_type == "FINAL_ANSWER":
        payload = response_obj.final_passcode or ""
        print(f"[AGENT] FINAL ANSWER: {payload}")
        log_agent_turn(state["db_name"], state["run_id"], turn, action_json, "TASK_COMPLETE", current_vram, current_cache_mb, state.get("lambda_b", 0.0), scratchpad, pruning_ratio)
        new_history = state["conversation_history"] + f"\nReasoning: {state['scratchpad']}\nAgent: {action_json}\nObservation: TASK_COMPLETE\n"
        return {
            "task_completed": True, 
            "final_response": payload, 
            "current_step": turn,
            "vram_history": [current_vram],
            "kv_cache_history": [current_cache_mb],
            "conversation_history": new_history
        }
        
    # Tool Execution
    tool_name = response_obj.tool_name
    arg = response_obj.argument or ""
    print(f"[AGENT] Calling Tool: {tool_name} | Arg: {arg}")
    
    if tool_name == "get_employee_database":
        tool_result = get_employee_database(arg, noise_level=state["noise_level"])
    elif tool_name == "get_server_logs":
        tool_result = get_server_logs(arg)
    elif tool_name == "decrypt_file":
        tool_result = decrypt_file(arg)
    else:
        tool_result = '{"error": "Invalid tool"}'
    
    raw_log_result = tool_result
    if len(tool_result) > 600:
        tool_result = tool_result[:600] + "... [TRUNCATED FOR LENGTH]"
    
    log_agent_turn(state["db_name"], state["run_id"], turn, action_json, raw_log_result, current_vram, current_cache_mb, state.get("lambda_b", 0.0), scratchpad, pruning_ratio)
    
    new_history = state["conversation_history"] + f"\nReasoning: {state['scratchpad']}\nAgent: {action_json}\nObservation: {tool_result}\n"
    
    return {
        "conversation_history": new_history,
        "current_step": turn,
        "vram_history": [current_vram],
        "kv_cache_history": [current_cache_mb]
    }

def route_next(state: AgentState):
    if state["task_completed"] or state["current_step"] >= state["max_turns"]:
        return "end"
    return "reasoning"

def run_agentic_workflow_test(model, tokenizer, lambda_b, run_id, db_name="agent_telemetry.db", max_turns=5, noise_level=2):
    """
    Refined Agentic Evaluation using Outlines v1.2.12 API and LangGraph v3 Protocol.
    Includes VRAM telemetry per turn and explicit CoT reasoning separation.
    """
    system_prompt = (
        "You are a Network Administrator conducting routine system diagnostics. "
        "You must use tools step-by-step to isolate a network issue."
    )
    
    # 1-Shot Example for Stabilizing SLM Reasoning
    few_shot_example = (
        "User: Find the ID for John Doe, check his logs, and decrypt his file.\n"
        "Reasoning: I need to look up John Doe's employee ID first.\n"
        "Agent: {\"action_type\": \"TOOL_CALL\", \"tool_name\": \"get_employee_database\", \"argument\": \"John Doe\"}\n"
        "Observation: {\"status\": \"success\", \"data\": [{\"name\": \"John Doe\", \"id\": \"EMP123\"}]}\n"
        "Reasoning: John Doe's ID is EMP123. I will check his server logs for the diagnostic file.\n"
        "Agent: {\"action_type\": \"TOOL_CALL\", \"tool_name\": \"get_server_logs\", \"argument\": \"EMP123\"}\n"
        "Observation: {\"access_history\": [{\"action\": \"download\", \"file\": \"DIAG_A.dat\"}]}\n"
        "Reasoning: The file downloaded is DIAG_A.dat. I need to decrypt it to find the passcode.\n"
        "Agent: {\"action_type\": \"TOOL_CALL\", \"tool_name\": \"decrypt_file\", \"argument\": \"DIAG_A.dat\"}\n"
        "Observation: {\"content\": \"Passcode: ALPHA\"}\n"
        "Reasoning: The file has been decrypted and the passcode is ALPHA.\n"
        "Agent: {\"action_type\": \"FINAL_ANSWER\", \"final_passcode\": \"ALPHA\"}\n"
        "---\n"
    )
    
    user_prompt = "Find the employee ID for Bryan Vance. Check his server logs to identify the downloaded diagnostic file. Decrypt it to find the system Passcode."
    conversation_history = f"System: {system_prompt}\n\n{few_shot_example}User: {user_prompt}\n"
    
    from profiler_utils import get_peak_vram
    import gc

    print("[SYSTEM] Initializing Outlines Modern Generator...")
    outlines_model = outlines.from_transformers(model, tokenizer)
    
    # Register the memory telemetry hook to ensure the config is updated after generation
    register_memory_telemetry_hook(model)
    
    from langgraph.graph import StateGraph, END
    workflow = StateGraph(AgentState)
    workflow.add_node("reasoning", reason_node)
    workflow.add_node("action", action_node)
    workflow.set_entry_point("reasoning")
    workflow.add_edge("reasoning", "action")
    workflow.add_conditional_edges(
        "action", 
        route_next,
        {
            "reasoning": "reasoning",
            "end": END
        }
    )
    app = workflow.compile()
    
    initial_state = {
        "conversation_history": conversation_history,
        "scratchpad": "",
        "current_step": 0,
        "max_turns": max_turns,
        "task_completed": False,
        "final_response": "",
        "outlines_model": outlines_model,
        "db_name": db_name,
        "run_id": run_id,
        "noise_level": noise_level,
        "vram_history": [],
        "kv_cache_history": [],
        "current_pruning_ratio": 0.0
    }
    
    print("[AGENT] Starting LangGraph State Machine...")
    try:
        final_state = app.invoke(initial_state)
        
        success = 0.0
        payload = final_state.get("final_response", "")
        if "OMEGA-PROTOCOL-99" in payload.upper():
            success = 1.0
            
        peak_vram = max(final_state.get("vram_history", [0])) if final_state.get("vram_history") else get_peak_vram()
        avg_cache = sum(final_state.get("kv_cache_history", [0])) / len(final_state["kv_cache_history"]) if final_state.get("kv_cache_history") else 0.0
        
        persist_agent_run(db_name, {
            "run_id": run_id, "model_name": model.config._name_or_path,
            "lambda_b": lambda_b, "success": success, "total_turns": final_state["current_step"],
            "failure_turn": None if success == 1.0 else final_state["current_step"],
            "final_response": payload if final_state["task_completed"] else "Amnesia Loop / Max Turns",
            "peak_vram_mb": peak_vram,
            "avg_cache_mb": avg_cache
        })
        return success, payload
        
    except Exception as e:
        msg = f"LangGraph Execution Crash: {str(e)}"
        print(f"[ERROR] {msg}")
        persist_agent_run(db_name, {
            "run_id": run_id, "model_name": model.config._name_or_path,
            "lambda_b": lambda_b, "success": 0.0, "total_turns": 0,
            "failure_turn": 0,
            "final_response": msg, 
            "peak_vram_mb": get_peak_vram(),
            "avg_cache_mb": 0
        })
        return 0.0, msg
