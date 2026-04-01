# repro_basic_agent.py: Self-Contained Agentic Evaluation Reproduction
import os
import sys
import subprocess
import argparse

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
            print(f"[WARNING] Project path {project_path} not found.")

        # 2. Hugging Face Login
        from huggingface_hub import login
        try:
            hf_token = userdata.get('HF_TOKEN')
            login(token=hf_token)
            os.environ["HF_TOKEN"] = hf_token
            print("[SYSTEM] Logged in to Hugging Face Hub.")
        except Exception as e:
            print(f"[WARNING] Could not retrieve HF_TOKEN: {e}")
    except ImportError:
        print("[SYSTEM] Local environment detected.")

def run_agent_eval(model_name: str, lambda_b: float = 0.0):
    """
    Executes the REAL agentic evaluation using the project's core utilities.
    No mocking.
    """
    print(f"==========================================")
    print(f">> TARGET MODEL: {model_name}")
    print(f">> PROTOCOL: λB = {lambda_b} (Agentic Stress Test)")
    print(f"==========================================")

    # Ensure SQL environment is set up
    print("[SYSTEM] Initializing SQL Live Environment...")
    subprocess.run(["python3", "sql_eval_setup.py"], check=True)
    subprocess.run(["python3", "init_agent_db.py", "--db", "repro_agent.db"], check=True)

    # Call the main benchmark script with --agent_test
    cmd = [
        "python3", "run_benchmarks.py",
        "--model", model_name,
        "--lambda_b", str(lambda_b),
        "--agent_test",
        "--agent_db", "repro_agent.db",
        "--run_id", f"repro_{model_name.replace('/', '_')}_lb{lambda_b}"
    ]
    
    print(f"[SYSTEM] Executing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    setup_colab_environment()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="google/gemma-2-2b-it")
    parser.add_argument("--lambda_b", type=float, default=0.0)
    args = parser.parse_args()
    
    # By default, we run at lambda_b=0.0 to reproduce the v1-v3 "Amnesia Loops" 
    # which occurred even without KV pruning in smaller models.
    run_agent_eval(args.model, args.lambda_b)
