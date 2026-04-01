import torch

def test(a_shape, b_shape):
    try:
        a = torch.ones(a_shape)
        b = torch.ones(b_shape)
        c = a + b
        print(f"Success: {a_shape} + {b_shape} -> {c.shape}")
    except Exception as e:
        print(f"Error for {a_shape} + {b_shape}: {e}")

# What if a=(1, 32, 1, 321) and b=(1, 1, 319, 321)
test((1, 32, 1, 321), (1, 1, 319, 321))

# What if a=(1, 32, 1, 319) and b=(1, 1, 321, 319)
test((1, 32, 1, 319), (1, 1, 321, 319))

# What if a=(1, 32, 321, 321) and b=(1, 1, 319, 319)
test((1, 32, 321, 321), (1, 1, 319, 319))
