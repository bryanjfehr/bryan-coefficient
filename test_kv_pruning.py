import unittest
import torch
from surgery_utils import prune_kv_cache

class TestKVPruning(unittest.TestCase):
    def test_prune_kv_cache(self):
        # batch=1, heads=1, seq_len=5, dim=2
        key_states = torch.randn((1, 1, 5, 2))
        value_states = torch.randn((1, 1, 5, 2))
        
        # attn_weights: (batch, heads, q_len=1, k_len=5)
        # Scores: [0.1, 0.01, 0.5, 0.001, 0.2]
        attn_weights = torch.tensor([[[[0.1, 0.01, 0.5, 0.001, 0.2]]]])
        
        lambda_b = 0.05
        sink_count = 2 # Keep indices 0 and 1 regardless of score
        
        # Scoring logic:
        # idx 0: Sink -> KEEP
        # idx 1: Sink -> KEEP
        # idx 2: 0.5 > 0.05 -> KEEP
        # idx 3: 0.001 < 0.05 -> EVICT
        # idx 4: 0.2 > 0.05 -> KEEP
        
        keep_indices = prune_kv_cache(attn_weights, lambda_b, sink_count)
        
        # Expected indices: [0, 1, 2, 4]
        self.assertEqual(len(keep_indices), 4)
        self.assertTrue(torch.equal(keep_indices, torch.tensor([0, 1, 2, 4])))
        
        # Verify physical surgery would work
        pruned_k = key_states[:, :, keep_indices, :]
        self.assertEqual(pruned_k.shape[2], 4)
        torch.testing.assert_close(pruned_k[:, :, 0, :], key_states[:, :, 0, :])

if __name__ == '__main__':
    unittest.main()
