import torch
try:
    a = torch.ones(1, 32, 319, 319)
    b = torch.ones(1, 1, 321, 319)
    c = a + b
except Exception as e:
    print("Error:", e)
