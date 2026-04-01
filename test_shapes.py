import torch

attn_weights = torch.ones(1, 32, 1, 319)
mask32 = torch.ones(1, 1, 1, 321)

try:
    res = attn_weights + mask32
except Exception as e:
    print("Error 1:", e)
    
attn_weights = torch.ones(1, 32, 1, 321)
mask32 = torch.ones(1, 1, 1, 319)

try:
    res = attn_weights + mask32
except Exception as e:
    print("Error 2:", e)
