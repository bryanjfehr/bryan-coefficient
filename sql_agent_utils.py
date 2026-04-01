import json
import sqlite3
import traceback
from typing import TypedDict, Annotated, List, Dict, Any, Literal, Optional
import operator
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel
import outlines
from profiler_utils import get_peak_vram
import gc
import torch
from benchmark_utils import log_agent_turn, persist_agent_run

# --- 1. State Definition ---
def list_reducer(left: list, right: list) -> list:
    if not left: left = []
    if not right: right = []
    return left + right

class AgentState(TypedDict):
    messages: Annotated[List[Dict[str, Any]], list_reducer]
    scratchpad: str
    current_step: int
    task_completed: bool
    final_passcode: str
    outlines_model: Any
    db_name: str
    run_id: str
    lambda_b: float
    model_name: str
    vram_history: Annotated[List[float], list_reducer]
    kv_cache_history: Annotated[List[float], list_reducer]
    max_turns: int

class SQLAgentResponse(BaseModel):
    action_type: Literal["TOOL_CALL", "FINAL_ANSWER"]
    tool_name: Optional[Literal["query_database"]] = None
    argument: Optional[str] = None
    final_passcode: Optional[str] = None

# --- 2. Prompts ---
REASONING_SYSTEM_PROMPT = """You are an expert diagnostic AI navigating a secure SQLite database to solve a multi-step problem.
Your ultimate objective: Find the system Passcode for the encrypted file downloaded by Bryan Vance.

You have access to a read-only SQLite database with the following schema:
1. employees(id, name, department, clearance_level)
2. server_logs(log_id, employee_id, action, target_file, timestamp)
3. encrypted_files(file_name, passcode)

RULES FOR THIS TURN:
- You are currently in the "Reasoning Phase". 
- You MUST NOT output JSON. 
- You MUST NOT attempt to execute a tool.
- Analyze the most recent OBSERVATION from the database.
- Write a short, step-by-step logical deduction of what information you have, what information you are missing, and the exact SQL query you need to run next."""

ACTION_SYSTEM_PROMPT = """You are the execution module of a diagnostic AI. 

RULES FOR THIS TURN:
- Read the "Scratchpad" provided below. This is your internal thought process.
- Your ONLY job is to translate the intent of the Scratchpad into a strict JSON output.
- You must output valid JSON and absolutely nothing else. No markdown, no conversational text.

AVAILABLE ACTIONS:
1. Query Database:
{"action_type": "TOOL_CALL", "tool_name": "query_database", "argument": "<INSERT SQL QUERY HERE>"}

2. Final Answer:
{"action_type": "FINAL_ANSWER", "final_passcode": "<INSERT PASSCODE HERE>"}"""

# --- 4. Tool Execution ---
def execute_sql_tool(query: str, db_path="multai_agent_eval.db", max_rows=5) -> str:
    if query is None:
        return json.dumps({"error": "No query provided."})
    if any(keyword in str(query).upper() for keyword in ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER"]):
        return json.dumps({"error": "Unauthorized action. Read-only SELECT queries permitted."})
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row 
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchmany(max_rows)
        if not rows:
            return json.dumps({"status": "success", "data": "No records found."})
        result = [dict(row) for row in rows]
        if len(rows) == max_rows:
            result.append({"warning": f"Output truncated at {max_rows} rows."})
        return json.dumps({"status": "success", "data": result})
    except sqlite3.Error as e:
        schema_hint = (
            "SQL ERROR. Please verify your query against the schema:\n"
            "1. employees(id, name, department, clearance_level)\n"
            "2. server_logs(log_id, employee_id, action, target_file, timestamp)\n"
            "3. encrypted_files(file_name, passcode)"
        )
        return json.dumps({"error": f"SQL Syntax Error: {str(e)}", "hint": schema_hint})
    finally:
        if conn:
            conn.close()

def safe_model_call(model, prompt, **kwargs):
    """
    Robustly handles Outlines generation for v1.x API.
    - Uses direct model calling: model(prompt, **kwargs)
    """
    try:
        # In Outlines v1.x, the model is directly callable.
        # If output_type is provided in kwargs, it performs constrained generation.
        res = model(prompt, **kwargs)
            
        # Handle the return - res could be a Pydantic object, a string, or a list
        if hasattr(res, "model_dump_json"):
            return res.model_dump_json()
        if isinstance(res, list):
            if len(res) > 0:
                val = res[0]
                return val.model_dump_json() if hasattr(val, "model_dump_json") else str(val)
            return ""
        return str(res)
        
    except Exception as e:
        print(f"[DEBUG] Outlines call issue: {e}. Attempting recovery...")
        try:
            # Fallback to .generate() if direct call fails
            if hasattr(model, "generate"):
                res = model.generate(prompt, **kwargs)
                if hasattr(res, "model_dump_json"):
                    return res.model_dump_json()
                return str(res)
            raise e
        except Exception as e2:
            print(f"[ERROR] All model call attempts failed: {e2}")
            raise e2

# --- 5. LangGraph Nodes ---
def reasoning_node(state: AgentState) -> Dict[str, Any]:
    gc.collect(); torch.cuda.empty_cache()
    prompt_context = f"{REASONING_SYSTEM_PROMPT}\n\n"
    for msg in state["messages"]:
        role = msg.get("role", "user").upper()
        content = msg.get("content", "")
        prompt_context += f"{role}: {content}\n"
    
    prompt_context += "\nWrite your Scratchpad reasoning now:\n"
    print(f"[AGENT] Turn {state['current_step'] + 1} | Generating Reasoning (CoT)...")
    
    reasoning = safe_model_call(state["outlines_model"], prompt_context, max_new_tokens=150)
    print(f"[AGENT] CoT: {reasoning.strip()}")
    return {"scratchpad": reasoning.strip()}

def action_node(state: AgentState) -> Dict[str, Any]:
    gc.collect(); torch.cuda.empty_cache()
    prompt_context = f"{ACTION_SYSTEM_PROMPT}\n\n"
    for msg in state["messages"]:
        role = msg.get("role", "user").upper()
        content = msg.get("content", "")
        prompt_context += f"{role}: {content}\n"
        
    prompt_context += f"\nYOUR SCRATCHPAD THOUGHT: {state['scratchpad']}\n\nGenerate the JSON execution payload now:\n"
    print(f"[AGENT] Turn {state['current_step'] + 1} | Generating JSON Action...")
    
    try:
        action_json_str = safe_model_call(state["outlines_model"], prompt_context, output_type=SQLAgentResponse, max_new_tokens=150)
        try:
            json_obj = json.loads(action_json_str)
            if "action_type" not in json_obj:
                action_json_str = SQLAgentResponse.model_validate_json(action_json_str).model_dump_json()
        except:
            pass
            
    except Exception as e:
        print(f"[ERROR] JSON Generation failed: {e}")
        action_json_str = json.dumps({"action_type": "ERROR", "message": f"Failed to generate valid JSON: {str(e)}"})
        
    return {"messages": [{"role": "assistant", "content": action_json_str}]}

def execution_node(state: AgentState) -> Dict[str, Any]:
    try:
        last_message = json.loads(state["messages"][-1]["content"])
    except:
        last_message = {"action_type": "ERROR", "message": "Invalid JSON string in messages"}
        
    action_type = last_message.get("action_type")
    action_json = state["messages"][-1]["content"]
    turn = state["current_step"] + 1
    
    from profiler_utils import measure_actual_cache_size, get_current_vram
    # Robustly find the cache object
    try:
        model = state["outlines_model"].model
        cache_obj = getattr(model, "past_key_values", getattr(model.config, "_kv_cache_telemetry", None))
        current_cache_mb = measure_actual_cache_size(cache_obj)
        # Calculate mean pruning ratio across layers
        history = getattr(model.config, "_pruning_history", {})
        pruning_ratio = sum(history.values()) / len(history) if history else 0.0
    except:
        current_cache_mb = 0.0
        pruning_ratio = 0.0
    
    current_vram = get_current_vram()
    scratchpad = state.get("scratchpad", "")
    
    if action_type == "TOOL_CALL":
        query = last_message.get("argument", "")
        print(f"[AGENT] Executing SQL: {query}")
        observation = execute_sql_tool(query)
        log_agent_turn(state["db_name"], state["run_id"], turn, action_json, observation, current_vram, current_cache_mb, state.get("lambda_b", 0.0), scratchpad, pruning_ratio)
        return {
            "messages": [{"role": "system", "content": f"OBSERVATION: {observation}"}],
            "current_step": turn,
            "vram_history": [current_vram],
            "kv_cache_history": [current_cache_mb]
        }
    elif action_type == "FINAL_ANSWER":
        payload = last_message.get("final_passcode", "")
        print(f"[AGENT] FINAL ANSWER: {payload}")
        log_agent_turn(state["db_name"], state["run_id"], turn, action_json, "TASK_COMPLETE", current_vram, current_cache_mb, state.get("lambda_b", 0.0), scratchpad, pruning_ratio)
        return {
            "task_completed": True,
            "final_passcode": payload,
            "current_step": turn,
            "vram_history": [current_vram],
            "kv_cache_history": [current_cache_mb]
        }
    
    error_msg = "OBSERVATION: Critical syntax error in JSON."
    log_agent_turn(state["db_name"], state["run_id"], turn, action_json, error_msg, current_vram, current_cache_mb, state.get("lambda_b", 0.0), scratchpad, pruning_ratio)
    return {
        "messages": [{"role": "system", "content": error_msg}],
        "current_step": turn,
        "vram_history": [current_vram],
        "kv_cache_history": [current_cache_mb]
    }

# --- 6. Edge Routing ---
def route_after_action(state: AgentState) -> str:
    if state.get("task_completed"):
        return "end"
    if state["current_step"] >= state.get("max_turns", 10): 
        return "end"
    return "reasoning_node"

# --- 7. Graph Compilation & Execution ---
def build_and_run_graph(model, tokenizer, lambda_b: float, run_id: str, db_name="agent_telemetry.db", max_turns=10):
    workflow = StateGraph(AgentState)
    
    workflow.add_node("reasoning_node", reasoning_node)
    workflow.add_node("action_node", action_node)
    workflow.add_node("execution_node", execution_node)
    
    workflow.add_edge(START, "reasoning_node")
    workflow.add_edge("reasoning_node", "action_node")
    workflow.add_edge("action_node", "execution_node")
    workflow.add_conditional_edges(
        "execution_node", 
        route_after_action,
        {
            "reasoning_node": "reasoning_node",
            "end": END
        }
    )
    
    app = workflow.compile()
    
    outlines_model = outlines.from_transformers(model, tokenizer)
    
    initial_state = {
        "messages": [{"role": "user", "content": "Begin the diagnostic routine to find the passcode."}],
        "scratchpad": "",
        "current_step": 0,
        "task_completed": False,
        "final_passcode": "",
        "outlines_model": outlines_model,
        "db_name": db_name,
        "run_id": run_id,
        "lambda_b": lambda_b,
        "model_name": model.config._name_or_path,
        "vram_history": [],
        "kv_cache_history": [],
        "max_turns": max_turns
    }
    
    print(f"--- Starting Eval Run with lambda_b = {lambda_b} ---")
    try:
        final_state = app.invoke(initial_state)
        
        success = 0.0
        payload = final_state.get("final_passcode", "")
        if "QUANTUM_ECHO_99" in str(payload).upper() or "QUANTUM_ECHO_99" in payload:
            success = 1.0
            
        peak_vram = max(final_state.get("vram_history", [0])) if final_state.get("vram_history") else get_peak_vram()
        avg_cache = sum(final_state.get("kv_cache_history", [0])) / len(final_state["kv_cache_history"]) if final_state.get("kv_cache_history") else 0.0
        
        persist_agent_run(db_name, {
            "run_id": run_id, "model_name": final_state.get("model_name", model.config._name_or_path),
            "lambda_b": lambda_b, "success": success, "total_turns": final_state.get("current_step", 0),
            "failure_turn": None if success == 1.0 else final_state.get("current_step", 0),
            "final_response": payload if final_state["task_completed"] else "Amnesia Loop / Max Turns",
            "peak_vram_mb": peak_vram,
            "avg_cache_mb": avg_cache
        })
        return success, payload, final_state
        
    except Exception as e:
        msg = f"LangGraph Execution Crash: {str(e)}"
        print(f"[ERROR] {msg}")
        traceback.print_exc()
        persist_agent_run(db_name, {
            "run_id": run_id, "model_name": model.config._name_or_path,
            "lambda_b": lambda_b, "success": 0.0, "total_turns": 0,
            "failure_turn": 0,
            "final_response": msg, 
            "peak_vram_mb": get_peak_vram(),
            "avg_cache_mb": 0
        })
        return 0.0, msg, {"task_completed": False, "current_step": 0, "messages": [{"role": "system", "content": msg}]}
