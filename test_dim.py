import torch

# Test 1: Mismatch at dim 3 (last dim)
try:
    a = torch.ones(1, 32, 1, 331)
    b = torch.ones(1, 1, 1, 330)
    c = a + b
except Exception as e:
    print("Test 1:", e)

# Test 2: Mismatch at dim 2 (q_len)
try:
    a = torch.ones(1, 32, 1, 331)
    b = torch.ones(1, 1, 330, 331)
    c = a + b
except Exception as e:
    print("Test 2:", e)

# Test 3: Mismatch at dim 2 when b is 1, a is 331
try:
    a = torch.ones(1, 32, 331, 331)
    b = torch.ones(1, 1, 330, 331)
    c = a + b
except Exception as e:
    print("Test 3:", e)
