# repro_small_needle.py: Self-Contained Bryan Coefficient Evaluation (Real Surgery)
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

def run_needle_eval(model_name: str, lambda_b: float, max_chars: int = 8000):
    """
    Executes the REAL needle-in-a-haystack evaluation using the project's core utilities.
    No mocking.
    """
    print(f"==========================================")
    print(f">> TARGET MODEL: {model_name}")
    print(f">> PROTOCOL: λB = {lambda_b} (Needle-in-a-Haystack)")
    print(f"==========================================")

    # Call the main benchmark script
    cmd = [
        "python3", "run_benchmarks.py",
        "--model", model_name,
        "--lambda_b", str(lambda_b),
        "--max_chars", str(max_chars),
        "--run_id", f"repro_{model_name.replace('/', '_')}_lb{lambda_b}"
    ]
    
    print(f"[SYSTEM] Executing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    setup_colab_environment()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--lambda_b", type=float, default=0.001)
    parser.add_argument("--max_chars", type=int, default=8000)
    args = parser.parse_args()
    
    run_needle_eval(args.model, args.lambda_b, args.max_chars)
