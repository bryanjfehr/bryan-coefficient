# repro_small_needle.py: Self-Contained Bryan Coefficient Evaluation (Colab Ready)
import torch
import torch.nn as nn
import math
import sys
import os
import argparse
from types import MethodType
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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

# --- SURGERY UTILS ---
def patch_attention(module, lambda_b=0.01, eviction_threshold=1e-3, sink_protection=20):
    if not hasattr(module, "_original_forward"):
        module._original_forward = module.forward

    module.lambda_b = lambda_b
    module.eviction_threshold = eviction_threshold
    module.sink_protection = sink_protection
    module.local_window = 128
    
    # Simple architecture detection
    model_type = getattr(module.config, "model_type", "").lower()
    module._return_len = 2 if any(x in model_type for x in ["phi", "qwen", "gemma"]) else 3

    module.forward = MethodType(patched_forward, module)

def patched_forward(self, *args, **kwargs):
    # Simplified forward for reproduction
    hidden_states = args[0] if len(args) > 0 else kwargs.get("hidden_states")
    attention_mask = kwargs.get("attention_mask")
    past_key_values = kwargs.get("past_key_values", kwargs.get("past_key_value"))
    
    # 1. Projections
    q_proj = getattr(self, "q_proj", getattr(self, "query", None))
    k_proj = getattr(self, "k_proj", getattr(self, "key", None))
    v_proj = getattr(self, "v_proj", getattr(self, "value", None))
    
    head_dim = self.config.hidden_size // self.config.num_attention_heads
    num_heads = self.config.num_attention_heads
    num_kv_heads = getattr(self.config, "num_key_value_heads", num_heads)
    
    input_shape = hidden_states.shape[:-1]
    query_states = q_proj(hidden_states).view(*input_shape, num_heads, head_dim).transpose(1, 2)
    key_states = k_proj(hidden_states).view(*input_shape, num_kv_heads, head_dim).transpose(1, 2)
    value_states = v_proj(hidden_states).view(*input_shape, num_kv_heads, head_dim).transpose(1, 2)
    
    # 2. Cache Update (Simplified)
    if past_key_values is not None:
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, kwargs)

    # 3. Attention Calculation
    scaling = head_dim**-0.5
    num_groups = num_heads // num_kv_heads
    key_states_rep = key_states.repeat_interleave(num_groups, dim=1) if num_groups > 1 else key_states
    
    attn_weights = torch.matmul(query_states.to(torch.float32), key_states_rep.transpose(2, 3).to(torch.float32)) * scaling

    # 4. λB Penalty Injection
    lambda_b = getattr(self, "lambda_b", 0.0)
    if lambda_b > 0 and key_states.size(2) > self.sink_protection + self.local_window:
        q_len, kv_len = attn_weights.shape[-2], attn_weights.shape[-1]
        device = attn_weights.device
        q_pos = torch.arange(q_len, device=device).unsqueeze(1)
        kv_pos = torch.arange(kv_len, device=device).unsqueeze(0)
        past_len = kv_len - q_len
        distance = (q_pos + past_len) - kv_pos
        
        dist_mask = torch.ones((1, 1, 1, kv_len), device=device)
        dist_mask[..., 0:self.sink_protection] = 0.0
        dist_mask[..., -self.local_window:] = 0.0
        
        penalty = -1.0 * (lambda_b * distance.float() * dist_mask)
        attn_weights = attn_weights + penalty

    # 5. Softmax & Pruning
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
    
    if past_key_values is not None and lambda_b > 0:
        importance = attn_weights.sum(dim=(0, 1, 2))
        keep_mask = importance > self.eviction_threshold
        keep_mask[:self.sink_protection] = True
        keep_mask[-max(self.local_window, q_len):] = True
        
        survivors = torch.nonzero(keep_mask).squeeze(-1)
        if len(survivors) < key_states.size(2):
            # Physical Eviction
            past_key_values.key_cache[self.layer_idx] = key_states[:, :, survivors, :]
            past_key_values.value_cache[self.layer_idx] = value_states[:, :, survivors, :]
            attn_weights = attn_weights[:, :, :, survivors]
            attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-8)
            value_states_rep = value_states[:, :, survivors, :].repeat_interleave(num_groups, dim=1) if num_groups > 1 else value_states[:, :, survivors, :]
        else:
            value_states_rep = key_states_rep
    else:
        value_states_rep = value_states.repeat_interleave(num_groups, dim=1) if num_groups > 1 else value_states

    attn_output = torch.matmul(attn_weights.to(hidden_states.dtype), value_states_rep.to(hidden_states.dtype))
    attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1)
    
    o_proj = getattr(self, "o_proj", getattr(self, "dense", None))
    attn_output = o_proj(attn_output)

    return (attn_output, None, past_key_values) if self._return_len == 3 else (attn_output, past_key_values)

# --- EVALUATION SCRIPT ---
def run_eval(model_name, lambda_b, max_chars=8000):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto", attn_implementation="eager")

    # Patch layers
    for name, module in model.named_modules():
        if "Attention" in str(type(module)) or "attn" in name.lower():
            if hasattr(module, "q_proj") or hasattr(module, "query"):
                patch_attention(module, lambda_b=lambda_b)

    # Needle test
    haystack = "The secret passkey is 84729. " * (max_chars // 30)
    prompt = f"{haystack}\n\nQuestion: What is the secret passkey? Answer:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10)
    
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    success = "84729" in response
    print(f"[RESULT] lambda_b={lambda_b} | Success: {success} | Response: {response}")
    return success

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--lambda_b", type=float, default=0.001)
    args = parser.parse_args()
    run_eval(args.model, args.lambda_b)
