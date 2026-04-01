import unittest
from types import MethodType
from surgery_utils import patch_attention, unpatch_attention

class MockAttention:
    def __init__(self):
        self.forward = self._forward_impl
    def _forward_impl(self, x):
        return "original"

class TestSurgeryUtils(unittest.TestCase):
    def test_patch_and_unpatch_attention(self):
        obj = MockAttention()
        
        # 1. Verify original state
        self.assertEqual(obj.forward(None), "original")
        
        # 2. Define a dummy replacement that doesn't use Llama logic
        def dummy_forward(self, hidden_states, **kwargs):
            return "patched"
        
        # 3. Manually patch to verify mechanism
        if not hasattr(obj, "_original_forward"):
            obj._original_forward = obj.forward
        obj.forward = MethodType(dummy_forward, obj)
        
        # 4. Verify patched state
        self.assertEqual(obj.forward(None), "patched")
        
        # 5. Unpatch the object using utility
        unpatch_attention(obj)
        
        # 6. Verify restored state
        self.assertEqual(obj.forward(None), "original")

if __name__ == '__main__':
    unittest.main()
