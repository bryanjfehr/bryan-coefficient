import unittest
import torch
from surgery_utils import patch_attention, unpatch_attention

# Mocking the Attention classes
class LlamaAttention(torch.nn.Module):
    def forward(self, *args, **kwargs):
        return "original_llama"

class Qwen2Attention(torch.nn.Module):
    def forward(self, *args, **kwargs):
        return "original_qwen2"

class Phi3Attention(torch.nn.Module):
    def forward(self, *args, **kwargs):
        return "original_phi3"

class TestArchitecturePatching(unittest.TestCase):
    def test_llama_patching(self):
        module = LlamaAttention()
        patch_attention(module)
        # Check if the bound method is llama_patched_forward
        self.assertEqual(module.forward.__name__, "llama_patched_forward")
        
        # Mock attributes for coverage
        module.head_dim = 16
        module.num_key_value_groups = 1
        module.layer_idx = 0
        module.scaling = 1.0
        module.q_proj = torch.nn.Linear(64, 64)
        module.k_proj = torch.nn.Linear(64, 64)
        module.v_proj = torch.nn.Linear(64, 64)
        module.o_proj = torch.nn.Linear(64, 64)
        
        import surgery_utils
        with unittest.mock.patch('surgery_utils.get_transformers_module') as mock_get_mod:
            mock_mod = unittest.mock.MagicMock()
            mock_mod.apply_rotary_pos_emb = lambda q, k, c, s: (q, k)
            mock_mod.repeat_kv = lambda x, n: x
            mock_get_mod.return_value = mock_mod
            
            res, _ = module(torch.randn(1, 1, 64), position_embeddings=(None, None))
            self.assertIsInstance(res, torch.Tensor)

        unpatch_attention(module)
        self.assertEqual(module.forward(), "original_llama")

    def test_qwen2_patching(self):
        module = Qwen2Attention()
        patch_attention(module)
        self.assertEqual(module.forward.__name__, "qwen2_patched_forward")
        
        # Mock attributes for coverage
        module.head_dim = 16
        module.num_key_value_groups = 1
        module.layer_idx = 0
        module.scaling = 1.0
        module.q_proj = torch.nn.Linear(64, 64)
        module.k_proj = torch.nn.Linear(64, 64)
        module.v_proj = torch.nn.Linear(64, 64)
        module.o_proj = torch.nn.Linear(64, 64)
        
        import surgery_utils
        with unittest.mock.patch('surgery_utils.get_transformers_module') as mock_get_mod:
            mock_mod = unittest.mock.MagicMock()
            mock_mod.apply_rotary_pos_emb = lambda q, k, c, s: (q, k)
            mock_mod.repeat_kv = lambda x, n: x
            mock_get_mod.return_value = mock_mod
            
            res, _ = module(torch.randn(1, 1, 64), position_embeddings=(None, None))
            self.assertIsInstance(res, torch.Tensor)

        unpatch_attention(module)
        self.assertEqual(module.forward(), "original_qwen2")

    def test_phi3_patching(self):
        module = Phi3Attention()
        patch_attention(module)
        self.assertEqual(module.forward.__name__, "phi3_patched_forward")
        
        # Mock attributes for coverage
        module.head_dim = 16
        module.config = unittest.mock.MagicMock()
        module.config.num_attention_heads = 4
        module.config.num_key_value_heads = 4
        module.num_key_value_groups = 1
        module.layer_idx = 0
        module.scaling = 1.0
        module.qkv_proj = torch.nn.Linear(64, 64*3)
        module.o_proj = torch.nn.Linear(64, 64)
        
        import surgery_utils
        with unittest.mock.patch('surgery_utils.get_transformers_module') as mock_get_mod:
            mock_mod = unittest.mock.MagicMock()
            mock_mod.apply_rotary_pos_emb = lambda q, k, c, s: (q, k)
            mock_mod.repeat_kv = lambda x, n: x
            mock_get_mod.return_value = mock_mod
            
            res, _ = module(torch.randn(1, 1, 64), position_embeddings=(None, None))
            self.assertIsInstance(res, torch.Tensor)

        unpatch_attention(module)
        self.assertEqual(module.forward(), "original_phi3")

if __name__ == '__main__':
    unittest.main()
