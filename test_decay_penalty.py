import unittest
import torch
from surgery_utils import apply_decay_penalty

class TestDecayPenalty(unittest.TestCase):
    def test_apply_decay_penalty(self):
        # batch=1, heads=1, q_len=2, k_len=2
        # attn_weights: (batch, heads, q_len, k_len)
        attn_weights = torch.zeros((1, 1, 2, 2))
        lambda_b = 0.1
        
        # Expected distances for second query (row 1):
        # j=0: distance = 1
        # j=1: distance = 0
        
        patched_weights = apply_decay_penalty(attn_weights, lambda_b)
        
        # If penalty is subtractive: patched = original - lambda_b * distance
        # Row 1, Col 0: 0 - 0.1 * 1 = -0.1
        # Row 1, Col 1: 0 - 0.1 * 0 = 0
        
        self.assertAlmostEqual(patched_weights[0, 0, 1, 0].item(), -0.1)
        self.assertAlmostEqual(patched_weights[0, 0, 1, 1].item(), 0.0)
        
        # Row 0, Col 0: 0 - 0.1 * 0 = 0
        self.assertAlmostEqual(patched_weights[0, 0, 0, 0].item(), 0.0)

if __name__ == '__main__':
    unittest.main()
