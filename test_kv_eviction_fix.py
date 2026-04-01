import unittest
import torch
from types import SimpleNamespace
from surgery_utils import prune_kv_cache, llama_patched_forward

class TestKVEvictionFix(unittest.TestCase):
    def test_sink_expansion_50(self):
        """
        Verify that sink_count defaults to 50 and protects first 50 tokens.
        """
        # Create attention scores for 100 tokens. 
        # Tokens 0-49: low scores (below lambda_b)
        # Tokens 50-99: high scores (above lambda_b)
        # attn_weights shape: (batch, heads, q_len=1, k_len=100)
        attn_weights = torch.zeros((1, 1, 1, 100))
        attn_weights[:, :, :, 50:] = 0.5
        attn_weights[:, :, :, :50] = 0.001 
        
        lambda_b = 0.01
        sink_count = 50 
        
        keep_indices = prune_kv_cache(attn_weights, lambda_b, sink_count)
        
        # Should keep all 100 tokens (50 from sink, 50 from high scores)
        self.assertEqual(len(keep_indices), 100)
        
        # Test with even higher lambda_b
        lambda_b = 0.6
        keep_indices = prune_kv_cache(attn_weights, lambda_b, sink_count)
        # Should keep only 50 tokens (the sink)
        self.assertEqual(len(keep_indices), 50)
        self.assertTrue(torch.all(keep_indices < 50))

    def test_sink_count_default_50(self):
        """
        Verify that sink_count defaults to 50 in prune_kv_cache.
        """
        # Create attention scores for 100 tokens. 
        # Tokens 0-49: low scores (below lambda_b)
        # Tokens 50-99: high scores (above lambda_b)
        attn_weights = torch.zeros((1, 1, 1, 100))
        attn_weights[:, :, :, 50:] = 0.5
        attn_weights[:, :, :, :50] = 0.001 
        
        lambda_b = 0.1
        
        # Should keep at least 50 if default is 50
        keep_indices = prune_kv_cache(attn_weights, lambda_b)
        self.assertGreaterEqual(len(keep_indices), 50)
        self.assertEqual(len(keep_indices), 100)

    def test_physical_cache_slicing_real_dynamic_cache_mock(self):
        """
        Verify that llama_patched_forward physically slices DynamicCache 
        using the correct attributes/methods of HF DynamicCache.
        """
        # In HF DynamicCache, it stores key_cache and value_cache as lists of tensors.
        class MockAttention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.head_dim = 16
                self.num_heads = 4
                self.num_key_value_heads = 4
                self.num_key_value_groups = 1
                self.layer_idx = 0
                self.scaling = 1.0
                self.lambda_b = 0.1
                self.sink_count = 10
                self.q_proj = torch.nn.Linear(64, 64)
                self.k_proj = torch.nn.Linear(64, 64)
                self.v_proj = torch.nn.Linear(64, 64)
                self.o_proj = torch.nn.Linear(64, 64)
            
            def forward(self, *args, **kwargs):
                return llama_patched_forward(self, *args, **kwargs)

        class HFMockDynamicCache:
            def __init__(self):
                # Standard HF DynamicCache attributes
                self.key_cache = [] 
                self.value_cache = []
            def update(self, key_states, value_states, layer_idx):
                if len(self.key_cache) <= layer_idx:
                    self.key_cache.append(key_states)
                    self.value_cache.append(value_states)
                else:
                    self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
                    self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=2)
                return self.key_cache[layer_idx], self.value_cache[layer_idx]

        module = MockAttention()
        past_key_values = HFMockDynamicCache()
        
        # Add some initial tokens
        past_key_values.key_cache.append(torch.randn((1, 4, 100, 16)))
        past_key_values.value_cache.append(torch.randn((1, 4, 100, 16)))
        
        hidden_states = torch.randn((1, 1, 64))
        position_embeddings = (torch.ones((1, 101, 16)), torch.zeros((1, 101, 16)))
        
        import surgery_utils
        # Use patch with get_transformers_module to avoid real imports
        with unittest.mock.patch('surgery_utils.get_transformers_module') as mock_get_mod:
            mock_mod = unittest.mock.MagicMock()
            mock_mod.apply_rotary_pos_emb = lambda q, k, c, s: (q, k)
            mock_mod.repeat_kv = lambda x, n: x
            mock_get_mod.return_value = mock_mod
            
            with unittest.mock.patch('surgery_utils.prune_kv_cache', return_value=torch.arange(50)):
                output, weights = module(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    past_key_values=past_key_values
                )
                
                # Check if physical eviction happened
                self.assertEqual(past_key_values.key_cache[0].shape[2], 50)
                self.assertEqual(past_key_values.value_cache[0].shape[2], 50)

if __name__ == '__main__':
    unittest.main()
