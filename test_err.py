import torch

a = torch.ones(1, 32, 1, 321)
b = torch.ones(1, 1, 1, 319)
try:
    c = a + b
except Exception as e:
    print(f"Adding (..., 321) and (..., 319): {e}")

a = torch.ones(1, 32, 1, 319)
b = torch.ones(1, 1, 1, 321)
try:
    c = a + b
except Exception as e:
    print(f"Adding (..., 319) and (..., 321): {e}")
