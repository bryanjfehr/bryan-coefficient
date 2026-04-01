import torch
import argparse
import os
from types import MethodType
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Global dictionary to store intercepted attention weights
INTERCEPTED_ATTN_WEIGHTS = {}

def calculate_shannon_entropy(attn_weights):
    """
    Calculates Shannon Entropy of attention weights.
    attn_weights shape: (batch_size, num_query_heads, q_len, kv_len)
    """
    eps = 1e-12
    # Probability distribution over kv_len
    # sum(p * log(p)) across the key dimension
    entropy = -torch.sum(attn_weights * torch.log(attn_weights + eps), dim=-1)
    
    # Average across the batch and sequence length (queries)
    # Result: (num_query_heads,)
    # Note: dim=0 is batch, dim=1 is heads, dim=2 is q_len
    return entropy.mean(dim=(0, 2))

def create_intercept_forward(original_forward, layer_idx):
    """
    Wraps the native attention forward pass to intercept the `attn_weights`
    when `output_attentions=True` is passed.
    """
    def intercept_forward(self, *args, **kwargs):
        # Force the model to output attentions
        kwargs['output_attentions'] = True 
        
        # Call the original forward pass
        outputs = original_forward(*args, **kwargs)
        
        # In HF models, if output_attentions=True, the outputs tuple is usually:
        # (attn_output, attn_weights, past_key_value)
        # We search the tuple for a 4D tensor representing the attention probabilities
        attn_weights = None
        if isinstance(outputs, tuple):
            for item in outputs:
                if isinstance(item, torch.Tensor) and item.dim() == 4:
                    # Shape is (batch, heads, q_len, kv_len)
                    attn_weights = item
                    break
                
        if attn_weights is not None:
            # Calculate and store entropy immediately to save VRAM
            INTERCEPTED_ATTN_WEIGHTS[layer_idx] = calculate_shannon_entropy(attn_weights.detach()).cpu()
        else:
            print(f"[WARNING] Layer {layer_idx}: Could not intercept attention weights. Output length: {len(outputs) if isinstance(outputs, tuple) else 'N/A'}")
            
        return outputs
    return intercept_forward

def extract_entropy_matrix(model_id, text_path, max_chars=4000, alpha=2.0, lambda_base=0.001):
    print(f"[CALIBRATION] Starting entropy calibration for {model_id}...")
    
    hf_token = os.environ.get("HF_TOKEN")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="eager", # Crucial for intercepting raw weights
        token=hf_token
    )
    
    # Load calibration text
    with open(text_path, 'r', encoding='utf-8') as f:
        haystack = f.read()[:max_chars]
    
    inputs = tokenizer(haystack, return_tensors="pt").to(model.device)
    
    # --- SURGICAL INTERCEPTION SETUP ---
    num_layers = getattr(model.config, "num_hidden_layers", 
                         getattr(model.config, "num_layers", 
                                 getattr(model.config, "n_layers", 32))) # Fallback to 32 if not found
    
    # Re-verify via model structure if config is weird
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        num_layers = len(model.model.layers)
    elif hasattr(model, "layers"):
        num_layers = len(model.layers)
    
    # GQA awareness: We map lambda against the Query heads, as that dictates the attention matrix size
    num_q_heads = model.config.num_attention_heads 
    
    patched_count = 0
    for name, module in model.named_modules():
        is_attn = ("Attention" in module.__class__.__name__ or "attn" in name.lower()) and \
                  (hasattr(module, "q_proj") or hasattr(module, "qkv_proj"))
        
        if is_attn:
            import re
            # Extract layer index from name
            match = re.search(r'layers?\.(\d+)', name) or re.search(r'\.h\.(\d+)', name)
            if match:
                layer_idx = int(match.group(1))
                if layer_idx < num_layers:
                    # Monkey-patch the forward pass to intercept the weights
                    if not hasattr(module, "_original_forward"):
                        module._original_forward = module.forward
                    module.forward = MethodType(create_intercept_forward(module._original_forward, layer_idx), module)
                    patched_count += 1

    print(f"[CALIBRATION] Interceptors deployed on {patched_count} attention modules.")
    print(f"[CALIBRATION] Running unpenalized forward pass ({inputs['input_ids'].shape[1]} tokens)...")
    
    INTERCEPTED_ATTN_WEIGHTS.clear()
    
    with torch.no_grad():
        # Ensure output_attentions is propagated from the top level
        model(**inputs, output_attentions=True)
    
    # --- MATRIX CONSTRUCTION ---
    lambda_matrix = torch.zeros((num_layers, num_q_heads))
    
    if not INTERCEPTED_ATTN_WEIGHTS:
        print("[FATAL ERROR] Interception failed! No entropy data captured.")
        return lambda_matrix

    print(f"[SYSTEM] Constructing [ {num_layers} x {num_q_heads} ] λB tensor...")

    # Stack the 1D tensors from each layer into a 2D matrix
    try:
        # Check if all layers are present
        missing_layers = [i for i in range(num_layers) if i not in INTERCEPTED_ATTN_WEIGHTS]
        if missing_layers:
            print(f"[WARNING] Missing entropy data for layers: {missing_layers}. Filling with zeros.")
            for i in missing_layers:
                INTERCEPTED_ATTN_WEIGHTS[i] = torch.zeros(num_q_heads)
                
        all_entropies = torch.stack([INTERCEPTED_ATTN_WEIGHTS[i] for i in range(num_layers)])
    except Exception as e:
        print(f"[FATAL ERROR] Error stacking entropies: {e}")
        return lambda_matrix

    max_h = all_entropies.max()
    print(f"[CALIBRATION] Global Max Entropy (H_max): {max_h:.4f}")
    
    if max_h == 0:
        print("[WARNING] Max Entropy is 0. Entropy scaling will be uniform.")
        max_h = 1.0
    
    # Apply the mathematical mapping: λB,h,l = λB(base) * (1 - H/Hmax)^α
    for l in range(num_layers):
        h_l = INTERCEPTED_ATTN_WEIGHTS[l]
        lambda_matrix[l] = lambda_base * torch.pow(1.0 - (h_l / max_h), alpha)

    # --- CLEANUP ---
    for name, module in model.named_modules():
        if hasattr(module, "_original_forward"):
            module.forward = module._original_forward
            del module._original_forward

    return lambda_matrix

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--text", type=str, required=True, help="Path to the long-context calibration text.")
    parser.add_argument("--max_chars", type=int, default=4000)
    parser.add_argument("--lambda_base", type=float, required=True, help="The failure threshold from Phase 1.")
    parser.add_argument("--alpha", type=float, default=2.0, help="Aggression curve. 2.0 = steep protection for semantic heads.")
    parser.add_argument("--output", type=str, default="lambda_b_matrix.pt")
    args = parser.parse_args()
    
    matrix = extract_entropy_matrix(
        args.model, 
        args.text, 
        max_chars=args.max_chars, 
        alpha=args.alpha, 
        lambda_base=args.lambda_base
    )
    
    torch.save(matrix, args.output)
    print(f"[SUCCESS] Multidimensional λB matrix saved to {args.output}")

if __name__ == "__main__":
    main()
