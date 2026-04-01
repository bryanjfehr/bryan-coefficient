# repro_basic_agent.py: Self-Contained Agentic Evaluation Reproduction
import os
import sys
import sqlite3
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List, Union

# --- GOOGLE COLAB SETUP ---
def setup_colab_environment():
    try:
        from google.colab import drive, userdata
        print("[SYSTEM] Google Colab detected. Initializing environment...")
        
        # 1. Mount Drive
        drive.mount('/content/drive')
        project_path = '/content/drive/MyDrive/bryan-coefficient'
        if os.path.exists(project_path):
            os.chdir(project_path)
            if project_path not in sys.path:
                sys.path.append(project_path)
            print(f"[SYSTEM] Working directory set to: {os.getcwd()}")
        else:
            print(f"[WARNING] Project path {project_path} not found. Check your Drive folder name.")

        # 2. Hugging Face Login
        from huggingface_hub import login
        try:
            hf_token = userdata.get('HF_TOKEN')
            login(token=hf_token)
            print("[SYSTEM] Successfully logged in to Hugging Face Hub via Colab secrets.")
        except Exception as e:
            print(f"[WARNING] Could not retrieve HF_TOKEN from Colab secrets: {e}")
    except ImportError:
        print("[SYSTEM] Local environment detected. Skipping Colab-specific setup.")

setup_colab_environment()

# --- MOCK SQL SETUP ---
def setup_mock_db():
    conn = sqlite3.connect("mock_eval.db")
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT, email TEXT)")
    cursor.execute("INSERT OR IGNORE INTO users (id, name, email) VALUES (1, 'Alice', 'alice@example.com'), (2, 'Bob', 'bob@example.com')")
    conn.commit()
    conn.close()

def run_sql(query: str):
    conn = sqlite3.connect("mock_eval.db")
    cursor = conn.cursor()
    try:
        cursor.execute(query)
        res = cursor.fetchall()
        return str(res)
    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        conn.close()

# --- AGENT GRAPH ---
class AgentState(TypedDict):
    input: str
    chat_history: List[str]
    agent_outcome: Union[dict, None]
    steps: List[str]

def call_model(state: AgentState):
    # This is a placeholder for actual LLM call logic
    print(f"[AGENT] Input: {state['input']}")
    return {"agent_outcome": {"action": "sql_query", "action_input": "SELECT * FROM users"}}

def execute_tools(state: AgentState):
    outcome = state["agent_outcome"]
    if outcome["action"] == "sql_query":
        result = run_sql(outcome["action_input"])
        print(f"[TOOL] SQL Result: {result}")
        return {"steps": [f"SQL Result: {result}"]}
    return {"agent_outcome": None}

def run_repro_agent(model_name: str):
    print(f"[SYSTEM] Reproducing Agentic Evaluation for {model_name}...")
    setup_mock_db()
    
    # In a real Colab, we would load the model here.
    # For this script, we'll simulate a few turns.
    
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("action", execute_tools)
    
    workflow.set_entry_point("agent")
    workflow.add_edge("agent", "action")
    workflow.add_edge("action", END)
    
    app = workflow.compile()
    
    inputs = {"input": "List all users from the database."}
    for output in app.stream(inputs):
        for key, value in output.items():
            print(f"Node '{key}': Output: {value}")
    
    print("[SUCCESS] Agentic evaluation turn completed.")

if __name__ == "__main__":
    # In practice, provide a model path or use a mock
    run_repro_agent("Qwen/Qwen2.5-1.5B-Instruct")
