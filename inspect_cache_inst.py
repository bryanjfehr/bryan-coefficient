import torch
import transformers.cache_utils as cu

print("Testing Cache instantiation:")
dc = cu.DynamicCache()
print("DynamicCache dict:", dc.__dict__)

sc = cu.StaticCache(config=type("Config", (), {"hidden_size": 32, "num_attention_heads": 1, "num_hidden_layers": 1, "num_key_value_heads": 1})(), max_batch_size=1, max_cache_len=10, device="cpu", dtype=torch.float32)
print("StaticCache dict:", sc.__dict__)
